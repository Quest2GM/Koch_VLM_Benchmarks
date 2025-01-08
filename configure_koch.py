from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from zed import ZEDCamera

import subprocess
import cv2
import time
import numpy as np
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
            port="/dev/ttyACM0",
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

        # DH parameters
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
        
        self.robot_dh = DHRobot([RevoluteDH(a=A[i], 
                                            alpha=ALPHA[i], 
                                            d=D[i], 
                                            offset=OFFSET[i], 
                                            qlim=QLIM[i]) for i in range(len(A))])

        self.robot.connect()

        if torque:
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
        else:
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        
        print("Robot connected!")

        # Camera-robot calibration variables
        self.R, self.t = [], []


    def read_pos(self, iters):

        for _ in range(iters):

            # Follower arms outputs 6 positions because there are 6 motors
            robot_angles = self.robot.follower_arms["main"].read("Present_Position")

            # However, we only need the first five for DH - the last one is the end effector DOF
            robot_angles_rad = np.array(robot_angles)[:5] / 360 * 2 * np.pi

            # Between self.robot and self.robot_dh, the difference is that the second angle is flipped
            robot_angles_rad *= np.array([-1, -1, 1, 1, 1])

            out = self.robot_dh.fkine(robot_angles_rad)
            target_position = np.array(out)[:3, 3]
            print("Position:", target_position)

            # Verify correctness of inverse kinematics
            Q_after = self.inv_kin(target_position)
            print("Error:", np.linalg.norm(Q_after[:4]-robot_angles_rad[:4]))
            time.sleep(0.5)
            # robot_dh.plot(robot_angles_rad, block=True)
        
        return target_position


    def verify_inv(self, q, p, axis):
        limits = self.LLIM[axis]
        if np.isnan(q).any():
            return False
        if 0 <= axis <= 2 and not limits[0] <= p[axis] <= limits[1]:
            return False
        elif 4 <= axis <= 5 and not limits[0] <= q[axis] <= limits[1]:
            return False            
        return True


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
    

    def write_pos(self, q):
        q[1] *= -1  # switch first motor polarity
        q = q / (2*np.pi) * 360
        self.robot.follower_arms["main"].write("Goal_Position", q)


    def manual_control(self):
        
        curr_position = self.read_pos(iters=1)
        positions = []
        print("Current Position:", curr_position)

        def on_press(key):
            nonlocal positions

            # Position control
            pos_keys = ["1", "3", "5", "2", "4", "6"]
            pos_axis = [0, 1, 2, 0, 1, 2]
            m = 0.1
            pos_actions = [m, m, m, -m, -m, -m]

            # Gripper control
            grip_keys = ["7", "9", "8", "0"]
            grip_axis = [4, 5, 4, 5]
            m = 0.05
            grip_actions = [m, m, -m, -m]            

            if hasattr(key, 'char'):

                # Exit
                if key.char == "x":
                    listener.stop()
                    return positions
                
                # Position controller (q1, q2, q3)
                elif key.char in pos_keys:
                    i = pos_keys.index(key.char)
                    curr_position[pos_axis[i]] += pos_actions[i]
                    print("Position:", curr_position)
                    inv_kin = self.inv_kin(curr_position)
                    if self.verify_inv(inv_kin, curr_position, pos_axis[i]):
                        self.write_pos(inv_kin)
                        positions += [curr_position]
                    else:
                        print("Reached limit!")
                        curr_position[pos_axis[i]] -= pos_actions[i]

                # Gripper controller (q4, q5)
                elif key.char in grip_keys:
                    i = grip_keys.index(key.char)
                    axis = grip_axis[i]
                    if axis == 4:
                        self.q4 += grip_actions[i]
                    elif axis == 5:
                        self.q5 += grip_actions[i]
                    inv_kin = self.inv_kin(curr_position)
                    if self.verify_inv(inv_kin, curr_position, grip_axis[i]):
                        self.write_pos(inv_kin)
                    else:
                        print("Reached limit!")
                        if axis == 4:
                            self.q4 -= grip_actions[i]
                        elif axis == 5:
                            self.q5 -= grip_actions[i]

        print("Axis Controls: 2<-x->1 , 4<-y->3, 6<-z->5")
        print("Gripper Controls: 7<-z->8 , 9<-g->0")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        return positions


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
            curr_position = self.read_pos(iters=1)
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
                    keep_coords = input("Keep? (y/n)")
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
            iters = int(input("Number of read iterations?"))
            koch_robot.read_pos(iters)
        elif do == "m":
            koch_robot.manual_control()
        elif do == "c":
            koch_robot.camera_extrinsics(cam_node)
    except KeyboardInterrupt:
        koch_robot.exit()

    koch_robot.exit()
    

    
