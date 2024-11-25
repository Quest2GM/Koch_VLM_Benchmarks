from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import subprocess
import cv2
import time
import numpy as np
from pynput import keyboard
from roboticstoolbox import DHRobot, RevoluteDH

import pyzed.sl as sl

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
        self.PI = np.pi
        self.d1, self.a2, self.a3, self.d5 = 5.5, 20.83, 10, 10.5
        A = [0, self.a2, self.a3, 0, 0]
        ALPHA = [-self.PI/2, 0, 0, self.PI/2, 0]
        D = [self.d1, 0, 0, 0, self.d5]
        OFFSET = [0, 0, 0, self.PI/2, 0]
        self.LLIM = [[1,30], [-20, 20], []]
        QLIM = [[-self.PI/2, self.PI/2], [0, self.PI/2], [-self.PI/2, self.PI/2],\
                [-self.PI/2, self.PI/2], [-self.PI, self.PI]]
        
        self.robot_dh = DHRobot([RevoluteDH(a=A[i], 
                                            alpha=ALPHA[i], 
                                            d=D[i], 
                                            offset=OFFSET[i], 
                                            qlim=QLIM[i]) for i in range(len(A))])

        self.robot.connect()
        print("Robot connected!")

        if torque:
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
        else:
            self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Camera-robot calibration variables
        self.R, self.t = [], []


    def read_pos(self, iters):

        for _ in range(iters):

            # Follower arms outputs 6 positions because there are 6 motors
            robot_angles = self.robot.follower_arms["main"].read("Present_Position")

            # However, we only need the first five for DH - the last one is the end effector DOF
            robot_angles_rad = np.array(robot_angles)[:5] / 360 * 2 * self.PI

            # Between self.robot and self.robot_dh, the difference is that the second angle is flipped
            robot_angles_rad *= np.array([1, -1, 1, 1, 1])

            out = self.robot_dh.fkine(robot_angles_rad)
            target_position = np.array(out)[:3, 3]
            print("Position:", target_position)

            # Verify correctness of inverse kinematics
            Q_after = self.inv_kin(target_position)
            print("Error:", np.linalg.norm(Q_after[:4]-robot_angles_rad[:4]))
            time.sleep(0.5)
            # robot_dh.plot(robot_angles_rad, block=True)
        
        return target_position


    def verify_inv(self, q):
        return not np.isnan(q).any()


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
        phi_3 = self.PI - q3 - phi_4

        q4, q5 = self.PI/2, 0

        Q = [q1, -q2, q3, q4, q5, self.PI/2]

        return np.array(Q)
    
    def write_pos(self, q):
        q[1] *= -1  # switch first motor polarity
        q = q / (2*self.PI) * 360
        self.robot.follower_arms["main"].write("Goal_Position", q)


    def manual_control(self):
        
        curr_position = self.read_pos(iters=1)
        positions = []
        print("Current Position:", curr_position)

        def on_press(key):
            nonlocal positions

            keys = ["1", "3", "5", "2", "4", "6"]
            axis = [0, 1, 2, 0, 1, 2]
            m = 0.1
            actions = [m, m, m, -m, -m, -m]

            if hasattr(key, 'char'):
                if key.char == "x":
                    listener.stop()
                    return
                elif key.char in keys:
                    i = keys.index(key.char)
                    curr_position[axis[i]] += actions[i]
                    print("Position:", curr_position)
                    inv_kin = self.inv_kin(curr_position)
                    if self.verify_inv(inv_kin):
                        self.write_pos(inv_kin)
                        positions += [curr_position]
                    else:
                        print("Reached limit!")
                        curr_position[axis[i]] -= actions[i]

        print("Axis Controls: 2<-x->1 , 4<-y->3, 6<-z->5")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


    def camera_calibration(self, cam_node):
        
        # TODO: Find some way to auto generate this
        calibration_poses = [[6.729, -15.97, 26.7], [8.934, -23.48, 14.41], [15.94, -1.57, 24.91], [21.43, 19.3, 12.94], [4.938, 26.3, 24.84], [23.84, 13.5, 6.85]]
        calibration_coords = []

        for final_position in calibration_poses:
            
            # Generate positions from current position to end position
            curr_position = self.read_pos(iters=1)
            lin_x = np.linspace(curr_position[0], final_position[0], num=100)
            lin_y = np.linspace(curr_position[1], final_position[1], num=100)
            lin_z = np.linspace(curr_position[2], final_position[2], num=100)
            grad_poses = np.vstack((lin_x, lin_y, lin_z)).T
            inv_kin = [self.inv_kin(g) for g in grad_poses]

            # Move end effector to goal
            print("Sending to goal...")
            for I in inv_kin:
                time.sleep(0.02)
                self.write_pos(I)
            print("Reached!")
            time.sleep(1)   # wait before capturing picture

            # Show image to determine EE coordinates
            cv2.imshow("robot_img", cam_node.capture_image())
            cv2.waitKey(0)

            # TODO: Need to find some way to automate this
            x_in = input('x:')
            y_in = input('y:')
            calibration_coords += [[int(x_in), int(y_in)]]

            cv2.destroyAllWindows()

        calibration_poses = np.array(calibration_poses).reshape(-1,3).astype(np.float32)
        calibration_coords = np.array(calibration_coords).reshape(-1,2).astype(np.float32)

        # Estimate the rotation vector (rvec) and translation vector (tvec)
        _, rvec, self.t = cv2.solvePnP(calibration_poses, calibration_coords, cam_node.K, distCoeffs=None)
        self.R, _ = cv2.Rodrigues(rvec)


    def exit(self):
        input("Press return to deactivate robot...")
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.disconnect()



class ZEDCamera:

    def __init__(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        # Open the camera
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED Camera")
            exit(1)

        camera_info = self.zed.get_camera_information()
        self.image_zed = sl.Mat(camera_info.camera_configuration.resolution.width, 
                                camera_info.camera_configuration.resolution.height, 
                                sl.MAT_TYPE.U8_C4)
        calib_params = camera_info.camera_configuration.calibration_parameters
        self.left_cam_params = calib_params.left_cam

        self.K = self.get_K()
        self.D = self.get_D()


    def get_K(self):
        K = np.array([
            [self.left_cam_params.fx, 0, self.left_cam_params.cx],
            [0, self.left_cam_params.fy, self.left_cam_params.cy],
            [0, 0, 1]
        ])
        return K
    

    def get_D(self):
        D = np.array([
            self.left_cam_params.disto[0],  # k1
            self.left_cam_params.disto[1],  # k2
            self.left_cam_params.disto[4],  # k3
            self.left_cam_params.disto[2],  # p1 (tangential distortion)
            self.left_cam_params.disto[3],  # p2 (tangential distortion)
        ])
        return D
    
    def capture_image(self):

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            image_ocv = self.image_zed.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)

        return image_rgb


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
            koch_robot.camera_calibration(cam_node)
    except KeyboardInterrupt:
        koch_robot.exit()

    koch_robot.exit()
    

    
