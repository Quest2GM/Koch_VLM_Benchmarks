import torch
import numpy as np
import json
import os
import argparse
from rekep.environment import ReKepEnv
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.ik_solver import KochIKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
import rekep.transform_utils as T
from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

from camera import ZEDCamera
from vision_pipeline import SAM
from configure_koch import KochRobot

from openai import OpenAI
import ast, time

class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./rekep/configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize

        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])

        # Initialize robot and camera
        self.robot = KochRobot(port="/dev/ttyACM0", torque=True)
        self.camera = ZEDCamera()
        self.sam = SAM()

        # Get point to world conversion
        self.robot.get_point_to_world_conversion(self.camera)

        # initialize environment (real world)
        self.env = ReKepEnv(global_config['env'], self.robot, self.camera)

        # ik_solver
        ik_solver = KochIKSolver(self.robot)

        # initialize solvers
        reset_joint_pos = self.env.get_arm_joint_positions()
        self.subgoal_opt = False   # can choose to optimize subgoal or directly go to keypoint positions
        self.path_opt = False   # can choose to optimize path or directly interpolate to save time
        if self.subgoal_opt:
            self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, reset_joint_pos)
        if self.path_opt:
            self.path_solver = PathSolver(global_config['path_solver'], ik_solver, reset_joint_pos)
        
        # initialize visualizer
        self.visualizer = Visualizer(global_config['visualizer'], self.env)
        self.visualize = False

        # OpenAI client
        self.ai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        rgb = self.camera.capture_image("rgb")
        points = self.camera.pixel_to_3d_points()
        mask = self.sam.generate(rgb)

        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)


    def _get_all_subgoals(self, task_dir):

        # save prompt
        with open(os.path.join(task_dir, 'output_raw.txt'), 'r') as f:
            prompt_1 = f.read()

        prompt_2 = "Without providing any explanation, return an python integer \
                    list that has the keypoint indices that the end-effector needs to be at \
                    for each stage."
            
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_1 + ".\n" + prompt_2
                    },
                ]
            }
        ]

        stream = self.ai_client.chat.completions.create(model='gpt-4o',
                                                        messages=messages,
                                                        temperature=0.0,
                                                        max_tokens=2048,
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')

        output = output.replace("```", "").replace("python", "")

        return ast.literal_eval(output)


    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'], self.camera, rekep_program_dir)
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        if not self.subgoal_opt:
            subgoal_idxs = self._get_all_subgoals(rekep_program_dir)

        # main loop
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_positions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()

            # ====================================
            # = get optimized plan
            # ====================================
            curr_pose = self.robot.get_ee_pose()[0]
            print("Current pose:", curr_pose)
            if self.subgoal_opt:
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
            else:
                next_subgoal = np.concatenate([self.keypoints[subgoal_idxs[self.stage - 1] + 1], \
                                               curr_pose[3:]])
                if self.stage == 2:
                    next_subgoal[2] += 5
                subgoal_pose_homo = T.convert_pose_quat2mat(next_subgoal)
                next_subgoal[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 3, 0, -self.config['grasp_depth'] / 2.0])
            print("Next subgoal:", next_subgoal)

            # Optimize path, otherwise do direct interpolation
            if self.path_opt:
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
            else:
                num_points = 100
                next_path = np.zeros((num_points, 8))
                goal_lin = np.linspace(curr_pose, next_subgoal, num=num_points)
                next_path[:, :7] = goal_lin

            self.first_iter = False
            self.action_queue = next_path.tolist()

            # ====================================
            # = execute
            # ====================================
            # determine if we proceed to the next stage
            self.env.execute_action(self.action_queue)

            if self.is_grasp_stage:
                self._execute_grasp_action()
            elif self.is_release_stage:
                self._execute_release_action()
            
            # End condition
            if self.stage == self.program_info['num_stages']: 
                self.env.sleep(2.0)
                print("Finished!")
                return

            # progress to next stage
            self._update_stage(self.stage + 1)


    def _get_next_subgoal(self, from_scratch):
        print("getting next subgoal...")
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        print("getting next path...")
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        if stage == 1:
            self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        # grasp_pose = pregrasp_pose.copy()
        # grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([pregrasp_pose, [self.env.get_gripper_close_action()]])
        grasp_action = grasp_action.reshape(1, -1)
        self.env.execute_action(grasp_action, precise=True)
    
    def _execute_release_action(self):
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_red_pen.json',
            'instruction': 'pick up eraser and drop it into the masking tape',
            'rekep_program_dir': './rekep/vlm_query/2025-01-31_16-19-13_pick_up_eraser_and_drop_it_into_the_masking_tape'
            },
    }
    task = task_list['pen']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None)