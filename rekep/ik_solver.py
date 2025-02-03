"""
Adapted from OmniGibson and the Lula IK solver
"""

class IKResult:
    """Class to store IK solution results"""
    def __init__(self, success, joint_positions, error_pos, error_rot, num_descents=None):
        self.success = success
        self.cspace_position = joint_positions
        self.position_error = error_pos
        self.rotation_error = error_rot
        self.num_descents = num_descents if num_descents is not None else 1


class KochIKSolver:

    def __init__(self, robot):
        self.robot = robot

    def solve(self, target_pose_homo,
              position_tolerance=0.01,
              orientation_tolerance=0.05,
              max_iterations=150,
              initial_joint_pos=None):
        """
        IK solver for Koch robot to resemble ReKep IKSolver.
        """

        # Extract position and orientation
        target_pos = target_pose_homo[:3, 3]
        target_rot = target_pose_homo[:3, :3]   # ignore inverse orientation

        # Verify IK
        verify = self.robot.set_ee_pose(target_pos, axis=0, steps=0)
        if verify:
            q = self.robot.inv_kin(target_pos.flatten())
            return IKResult(success=True,
                            joint_positions=q,
                            error_pos=0,
                            error_rot=0,
                            num_descents=1)
        else:
            return IKResult(success=False,
                            joint_positions=initial_joint_pos,
                            error_pos=0,
                            error_rot=0,
                            num_descents=1)
        


