from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from zed import ZEDCamera

import subprocess
import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pynput import keyboard
from roboticstoolbox import DHRobot, RevoluteDH

import warnings
warnings.filterwarnings("ignore")


class KochRobot:

    def __init__(self, port, torque):

        # sudo enable robot USB port
        command = ["sudo", "chmod", "666", port]
        subprocess.run(command, check=True, text=True, capture_output=True)

        follower_arm = DynamixelMotorsBus(
            port=port,
            motors={
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )

        self.robot = ManipulatorRobot(
            robot_type="koch",
            follower_arms={"main": follower_arm},
            calibration_dir=".cache/calibration/koch",
        )
        self.robot.connect()
        print("Koch Connected!")

        # DH parameters and limits
        self.d1, self.a2, self.a3, self.d5 = 5.5, 10.68, 10, 10.5
        A = [0, self.a2, self.a3, 0, 0]
        ALPHA = [-np.pi/2, 0, 0, np.pi/2, 0]
        D = [self.d1, 0, 0, 0, self.d5]
        OFFSET = [0, 0, 0, np.pi/2, 0]
        QLIM = [[-np.pi/2, np.pi/2], [0, np.pi/2], [-np.pi/2, np.pi/2],\
                [-np.pi/2, np.pi/2], [-np.pi, np.pi]]
        self.LLIM = [[8, 17], [-15, 15], [1, 20], [], [-np.pi/2, np.pi/2], [0, np.pi/2]]
        
        # Initial EE angles
        self.q4, self.q5 = 0, np.pi / 2
        
        # For computing forward kinematics
        self.robot_dh = DHRobot([RevoluteDH(a=A[i], 
                                            alpha=ALPHA[i], 
                                            d=D[i], 
                                            offset=OFFSET[i], 
                                            qlim=QLIM[i]) for i in range(len(A))])

        # Initialize with torch enabled/disabled
        torque_mode = TorqueMode.ENABLED.value if torque else TorqueMode.DISABLED.value
        self.robot.follower_arms["main"].write("Torque_Enable", torque_mode)

        # Set to home position
        self.home_position = [8.2, 0, 5.5]
        self.set_to_home()

        time.sleep(1)
        self.curr_position = self.get_ee_pos()


    def get_arm_joint_angles(self):
        """
        Reads current joint angles of each revolute joint.
        """
        angles = self.robot.follower_arms["main"].read("Present_Position")
        angles = np.array(angles) /  360 * 2 * np.pi
        return angles


    def get_ee_pose(self):
        """
        Reads joint angles and computes ee pose.
        """

        # Follower arms outputs 6 positions because there are 6 motors
        robot_angles_full = self.get_arm_joint_angles()

        # However, we only need the first five for DH - the last one is gripper open/close
        robot_angles = np.array(robot_angles_full)[:5]

        # Between self.robot and self.robot_dh, the difference is that first and second angle is flipped
        # Note that first angle is flipped to ensure right-handed axis
        robot_angles *= np.array([-1, -1, 1, 1, 1])

        T = np.array(self.robot_dh.fkine(robot_angles))
        ee_matrix, ee_pos = T[:3, :3], T[:3, 3]
        quat_xyzw = R.from_matrix(ee_matrix).as_quat()

        # Verify correctness of inverse kinematics
        Q_from_inv = self.inv_kin(ee_pos)
        diff = np.linalg.norm(Q_from_inv[:4] - robot_angles[:4])
        # if diff > 0.01:
        #     raise Exception("Inverse kinematics computed incorrectly!", Q_from_inv, robot_angles)
        
        return np.hstack((ee_pos, quat_xyzw))

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]
    

    def inv_kin(self, target_position):

        px, py, pz = target_position

        # Analytic Solver
        r1 = np.sqrt(self.d5**2 + self.a3**2)
        r2 = np.sqrt(px**2 + py**2 + (pz-self.d1)**2)

        q1 = np.arctan2(py, px)

        phi_1 = np.arctan2(pz-self.d1, np.sqrt(px**2 + py**2))
        D2 = (self.a2**2 + r2**2 - r1**2) / (2*self.a2*r2)
        phi_2 = np.arctan2(np.sqrt(1-D2**2), D2)
        q2 = phi_1 + phi_2

        phi_4 = np.arctan2(self.d5, self.a3)
        D = (r2**2 - self.a2**2 - r1**2) / (2*self.a2*r1)
        q3 = np.arctan2(np.sqrt(1-D**2), D) - phi_4
        phi_3 = np.pi - q3 - phi_4

        Q = [-q1, -q2, q3, np.pi/2, self.q4, self.q5]

        return np.array(Q)
    
    
    def set_ee_pose(self, pos, axis):

        def verify_inv(q):
            limits = self.LLIM[axis]
            if np.isnan(q).any():
                return False
            if 0 <= axis <= 2 and not limits[0] <= pos[axis] <= limits[1]:
                return False
            elif 4 <= axis <= 5 and not limits[0] <= q[axis] <= limits[1]:
                return False            
            return True

        q = self.inv_kin(pos)
        if verify_inv(q):
            q[1] *= -1
            q = q / (2 * np.pi) * 360
            self.robot.follower_arms["main"].write("Goal_Position", q)
            return True
        else:
            return False

    def set_gripper_open(self):
        self.q5 = 0
        self.set_ee_pose(self.curr_position, axis=5)

    def set_gripper_close(self):
        self.q5 = np.pi / 2
        self.set_ee_pose(self.curr_position, axis=5)

    def set_to_home(self):
        self.set_ee_pose(self.home_position, axis=0)


    def manual_control(self):
        
        # increment
        m = 0.05
        # xyz = position, eeo = end-effector orientation, eeg = grip open/close
        control_dict = {"1": {"ctrl": "xyz", "axis": 0, "direc": 1},
                        "2": {"ctrl": "xyz", "axis": 0, "direc": -1},
                        "3": {"ctrl": "xyz", "axis": 1, "direc": 1},
                        "4": {"ctrl": "xyz", "axis": 1, "direc": -1},
                        "5": {"ctrl": "xyz", "axis": 2, "direc": 1},
                        "6": {"ctrl": "xyz", "axis": 2, "direc": -1},
                        "7": {"ctrl": "eeo", "axis": 4, "direc": 1},
                        "8": {"ctrl": "eeo", "axis": 4, "direc": -1},
                        "9": {"ctrl": "eeg", "axis": 5, "direc": 1},
                        "0": {"ctrl": "eeg", "axis": 5, "direc": -1},
                       }

        def on_press(key):

            if hasattr(key, 'char'):

                # Exit
                if key.char == "x":
                    listener.stop()
                    return
                
                elif key.char in control_dict:

                    # Get control parameters
                    control = control_dict[key.char]
                    ctrl, axis, direc = control["ctrl"], control["axis"], control["direc"]

                    # Set position
                    if ctrl == "xyz":
                        self.curr_position[axis] += direc * m
                    elif ctrl == "eeo":
                        self.q4 += direc * m
                    elif ctrl == "eeg":
                        self.set_gripper_open() if direc == 1 else self.set_gripper_close()

                    # Revert if command has reached limit
                    if not self.set_ee_pose(self.curr_position, axis):
                        if ctrl == "xyz":
                            self.curr_position[axis] -= direc * m
                        elif ctrl == "eeo":
                            self.q4 -= direc * m
                        print("Limit reached!")

                    print("Position:", self.curr_position)

        print("Axis Controls: 2<-x->1 , 4<-y->3, 6<-z->5")
        print("Gripper Controls: 7<-z->8 , 9<-g->0")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        return


    def camera_extrinsics(self, cam_node):

        # Pick up object for ee-tracking
        print("Pick up the object to track the end effector...")
        self.manual_control()
        
        # Hardcoded poses - we will capture all intermediate poses automatically
        move_poses = [[9, -13, 15], [16.74, -14.48, 5.1],\
                      [12.83, -1.57, 7], [9, 13, 15],
                      [17.46, 15, 8]]
        calibration_poses, calibration_coords = [], []

        for final_position in move_poses:
            
            # Generate positions from current position to end position
            curr_position = self.get_ee_pos()
            lin_x = np.linspace(curr_position[0], final_position[0], num=100)
            lin_y = np.linspace(curr_position[1], final_position[1], num=100)
            lin_z = np.linspace(curr_position[2], final_position[2], num=100)
            grad_poses = np.vstack((lin_x, lin_y, lin_z)).T
            inv_kin = [self.inv_kin(g) for g in grad_poses]

            # Move end effector to goal
            print("Sending to goal...")
            for i, I in enumerate(inv_kin):
                time.sleep(0.02)
                self.write_pos(I)

                # Every 10th step get calibration coordinates and poses
                if i % 10 == 0:
                    print(i)
                    time.sleep(0.5)
                    coords = cam_node.detect_end_effector()
                    #keep_coords = input("Keep? (y/n)")
                    keep_coords = "y"
                    if keep_coords == "y":
                        calibration_coords += [coords]
                        calibration_poses += [grad_poses[i]]

            print("Reached!")
            time.sleep(1)   # wait before capturing picture

        calibration_poses = np.array(calibration_poses).reshape(-1,3).astype(np.float32)
        calibration_coords = np.array(calibration_coords).reshape(-1,2).astype(np.float32)

        # Estimate the rotation vector (rvec) and translation vector (tvec)
        _, rvec, self.t = cv2.solvePnP(calibration_poses, calibration_coords, cam_node.K, distCoeffs=cam_node.D)
        self.R, _ = cv2.Rodrigues(rvec)

        # Save output
        np.save("camera_extrinsics.npy", np.array([self.R.reshape(-1), self.t], dtype=object))

        return [self.R, self.t]


    def exit(self):
        input("Press return to deactivate robot...")
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.disconnect()


if __name__ == "__main__":

    PORT = "/dev/ttyACM0"
    ENABLE_TORQUE = True

    koch_robot = KochRobot(port=PORT, torque=ENABLE_TORQUE)
    cam_node = ZEDCamera()

    try:
        do = input("Action (read=r, manual=m, calib=c):")
        if do == "r":
            koch_robot.get_ee_pose()
        elif do == "m":
            koch_robot.manual_control()
        elif do == "c":
            koch_robot.camera_extrinsics(cam_node)
    except KeyboardInterrupt:
        koch_robot.exit()

    koch_robot.exit()
    

    
