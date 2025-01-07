from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import subprocess
import cv2
import time
import numpy as np
from scipy.ndimage import label
from PIL import Image
from pynput import keyboard
from roboticstoolbox import DHRobot, RevoluteDH

import pyzed.sl as sl

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import matplotlib.pyplot as plt

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
        self.d1, self.a2, self.a3, self.d5 = 5.5, 20.83, 10, 10.5
        A = [0, self.a2, self.a3, 0, 0]
        ALPHA = [-np.pi/2, 0, 0, np.pi/2, 0]
        D = [self.d1, 0, 0, 0, self.d5]
        OFFSET = [0, 0, 0, np.pi/2, 0]
        QLIM = [[-np.pi/2, np.pi/2], [0, np.pi/2], [-np.pi/2, np.pi/2],\
                [-np.pi/2, np.pi/2], [-np.pi, np.pi]]
        self.LLIM = [[5, 17], [-20, 20], [9, 25], [], [-np.pi/2, np.pi/2], [0, np.pi/2]]
        
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

        Q = [q1, -q2, q3, np.pi/2, self.q4, self.q5]

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
        move_poses = [[6.729, -15.97, 26.7], [8.934, -23.48, 14.41],\
                      [15.94, -1.57, 24.91], [21.43, 19.3, 12.94],\
                      [4.938, 26.3, 24.84]]
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

        # SAM2
        sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"), apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)


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
    

    def capture_image(self, image_type):

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:

            if image_type == "rgb":
                self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            elif image_type == "depth":
                self.zed.retrieve_image(self.image_zed, sl.VIEW.DEPTH)
            else:
                raise Exception("Invalid image type!")

            image_ocv = self.image_zed.get_data()
            image = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)
            return image
        
        else:
            raise Exception("Could not get RGB image!")
    

    def hsv_limits(self, color):
        c = np.uint8([[color]])  # BGR values
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        hue = hsvC[0][0][0]  # Get the hue value

        # Handle red hue wrap-around
        if hue >= 165:  # Upper limit for divided red hue
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= 15:  # Lower limit for divided red hue
            lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        else:
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

        return lowerLimit, upperLimit
    

    def detect_end_effector(self):

        def keep_largest_blob(image):

            # Ensure the image contains only 0 and 255
            binary_image = (image == 255).astype(int)

            # Label connected components
            labeled_image, num_features = label(binary_image)

            # If no features, return the original image
            if num_features == 0:
                return np.zeros_like(image, dtype=np.uint8)

            # Find the largest component by its label
            largest_blob_label = max(range(1, num_features + 1), key=lambda lbl: np.sum(labeled_image == lbl))

            # Create an output image with only the largest blob
            output_image = (labeled_image == largest_blob_label).astype(np.uint8) * 255

            return output_image

        color = [158, 105, 16]

        # Get bounding box around object
        frame = self.capture_image("rgb")
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerLimit, upperLimit = self.hsv_limits(color=color)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask = keep_largest_blob(mask)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imwrite("calib.png", frame)

        return [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    

    def sam2_masks(self):
        image = self.capture_image("rgb")
        mask_dict = self.mask_generator.generate(image)
        masks = [mask_dict[i]["segmentation"] for i in range(len(mask_dict))]
        return masks


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
    

    
