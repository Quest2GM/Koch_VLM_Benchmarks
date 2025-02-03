import time
import numpy as np
import os

from rekep.utils import (
    bcolors,
    get_clock_time,
)

from openai import OpenAI
import ast

class ReKepEnv:
    def __init__(self, config, robot, camera, verbose=False):
        self.video_cache = []
        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']

        self.robot = robot
        self.camera = camera
        self.gripper_state = int(self.robot.q5 == 0)

        # OpenAI client
        self.ai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # ======================================
    # = exposed functions
    # ======================================

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """Get signed distance field"""
        print("Getting SDF voxels (mock data)")
        # Return mock SDF grid
        sdf_voxels = np.zeros((10, 10, 10))
        return sdf_voxels


    def _get_obj_idx_dict(self, task_dir):
        
        # save prompt
        with open(os.path.join(task_dir, 'output_raw.txt'), 'r') as f:
            self.prompt = f.read()

        with open(os.path.join(task_dir, '../object_idx_template.txt'), 'r') as f:
            self.prompt_2 = f.read()
            
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt + ".\n" + self.prompt_2
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
    
    def register_keypoints(self, keypoints, camera, rekep_dir):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        i.e. Associate keypoints with their respective objects. We technically only need to pay attention to objects that will move in the future.
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)

        self.keypoints = keypoints
        self._keypoint_registry = dict()

        idx_obj_map = self._get_obj_idx_dict(rekep_dir)

        # Set keypoint indices using index map from gpt-4o
        for i, k in enumerate(self.keypoints):
            obj = "none"

            if i in idx_obj_map.keys():
                obj = idx_obj_map[i]

            img_coord = self.robot.convert_world_to_point(camera, k)
            self._keypoint_registry[i] = {"object": obj, 
                                          "keypoint": k,
                                          "img_coord": img_coord,
                                          "on_grasp_coord": None,
                                          "is_grasped": False}
        

    def get_keypoint_positions(self):
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3)
        Given the registered keypoints, this function returns their current positions in the world frame.
        Keypoints are updated by taking the keypoints and updating them if the gripper is holding them or not.
        """

        keypoint_positions = []
        for idx, obj in self._keypoint_registry.items():

            if obj["is_grasped"]:

                # Determine difference between last ee_pose and grip_pose
                init_ee_point = obj["on_grasp_coord"]
                _, ee_point = self.robot.return_estimated_ee(self.camera)
                diff_point = [ee_point[0] - init_ee_point[0], ee_point[1] - init_ee_point[1]]
                
                # Update curr_pose of object
                obj_point = obj["img_coord"]
                curr_point = [obj_point[0] + diff_point[0], obj_point[1] + diff_point[1]]
                obj["img_coord"] = list(curr_point)
                obj["on_grasp_coord"] = list(ee_point)

                # Convert updated coord to camera
                obj["keypoint"] = self.robot.find_closest_point_to_world(obj["img_coord"])
            
            # NOTE: keypoint positions may be inaccurate due to noisy ZED camera
            keypoint_positions.append(obj["keypoint"])
        
        return np.array(keypoint_positions)


    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        # assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint_registry[keypoint_idx]["object"]


    def get_collision_points(self, noise=True):
        """Get collision points of gripper"""
        # Return mock collision points
        collision_points = np.random.rand(100, 3)
        return collision_points


    def reset(self):
        self.robot.set_to_home()

    def is_grasping(self, candidate_obj=None):
        
        # if the object is not graspable, then return False
        if candidate_obj == "none":
            return False
        else:
            for k in self._keypoint_registry:
                kp = self._keypoint_registry[k]
                if kp["object"] == candidate_obj:
                    _, ee_point = self.robot.return_estimated_ee(self.camera)
                    obj_point = kp["img_coord"]
                    dist = np.linalg.norm(np.array(ee_point) - np.array(obj_point))

                    # Object is being grasped if the end effector is closed and close enough
                    # to the object keypoint # NOTE: This is not foolproof but it works
                    grasped = (self.robot.q5 == np.pi / 2) & (dist < 20)
                    self._keypoint_registry[k]["is_grasped"] = grasped
                    self._keypoint_registry[k]["on_grasp_coord"] = ee_point
                    return grasped

    def get_ee_pose(self):
        return self.robot.get_ee_pose()[0]

    def get_ee_pos(self):
        return self.robot.get_ee_pos()

    def get_ee_quat(self):
        return self.robot.get_ee_quat()
    
    def get_arm_joint_positions(self):
        return self.robot.get_arm_joint_angles()

    def close_gripper(self):
        self.robot.set_gripper_close()

    def open_gripper(self):
        self.robot.set_gripper_open()

    def get_last_og_gripper_action(self):
        return self.last_og_gripper_action
    
    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def get_gripper_null_action(self):
        return 0.0
    
    def get_cam_obs(self):
        return self.camera.capture_image("rgb")
    
    def execute_action(
            self,
            action,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            # if precise:
            #     pos_threshold = 0.03
            #     rot_threshold = 3.0
            # else:
            #     pos_threshold = 0.10
            #     rot_threshold = 5.0
            # action = np.array(action).copy()
            # assert action.shape == (8,)
            # target_pose = action[:7]
            # gripper_action = action[7]

            # ======================================
            # = status and safety check
            # ======================================
            # if np.any(target_pose[:3] < self.bounds_min) \
            #      or np.any(target_pose[:3] > self.bounds_max):
            #     print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
            #     target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            # current_pose = self.get_ee_pose()
            # pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            # rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            # pos_is_close = pos_diff < self.interpolate_pos_step_size
            # rot_is_close = rot_diff < self.interpolate_rot_step_size
            # if pos_is_close and rot_is_close:
            #     self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
            #     pose_seq = np.array([target_pose])
            # else:
            #     num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
            #     pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
            #     self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            intermediate_pos_threshold = 0.10
            intermediate_rot_threshold = 5.0
            pose_seq = np.array(action)
            gripper_action = pose_seq[-1, -1]

            for pose in pose_seq[:-1]:
                # self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
                # print("setting pose:", pose[:3])
                self.robot.set_ee_pose(pose[:3], axis=0, steps=2)
            # move to the final pose with required precision
            pose = pose_seq[-1]

            # self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
            self.robot.set_ee_pose(pose[:3], axis=0, steps=2)
            # compute error
            # pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            pos_error, rot_error = 0, 0
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action
            # ======================================
            if gripper_action == self.get_gripper_open_action():
                self.open_gripper()
            elif gripper_action == self.get_gripper_close_action():
                self.close_gripper()
            elif gripper_action == self.get_gripper_null_action():
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            
            return pos_error, rot_error
    
    def sleep(self, seconds):
        time.sleep(seconds)
    