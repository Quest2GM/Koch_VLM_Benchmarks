from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import subprocess
import cv2
import time
import numpy as np
from PIL import Image as PILImage
from pynput import keyboard
from roboticstoolbox import DHRobot, RevoluteDH

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import threading
import rclpy
from rclpy.executors import SingleThreadedExecutor


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
        QLIM = [[-self.PI/2, self.PI/2], [0, self.PI/2], [-self.PI/2, self.PI/2],\
                [-self.PI/2, self.PI/2], [-self.PI, self.PI]]
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
        print("Current Position:", curr_position)
        exit_flag = False

        def on_press(key):
            global exit_flag, curr_position
            keys = ["q", "w", "e", "a", "s", "d"]
            axis = [0, 1, 2, 0, 1, 2]
            actions = [0.05, 0.05, 0.05, -0.05, -0.05, -0.05]

            if key.char == "x":
                exit_flag = True
            elif key.char in keys:
                i = keys.index(key.char)
                curr_position[axis[i]] += actions[i]
                print("Position:", curr_position)
                inv_kin = self.inv_kin(curr_position)
                self.write_pos(inv_kin)

        # Start the listener
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        while not exit_flag:
            pass
        listener.stop()

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
            cv2.imshow("robot_img", cam_node.curr_img)
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
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.disconnect()



class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # Create a subscriber on the input topic
        self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            '/zed/zed_node/rgb/image_rect_color',
            self.img_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.info_callback,
            10
        )

        self.curr_img = []
        self.K = []

    def img_callback(self, msg):

        # Handle different encoding formats
        if msg.encoding == 'rgb8':
            image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        elif msg.encoding == 'mono8':
            image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        elif msg.encoding == 'bgra8':
            image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
            image_array = image_array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
        elif msg.encoding == '32FC1':
            bridge = CvBridge()
            cv2_img = bridge.imgmsg_to_cv2(msg, '32FC1')
            image_array = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            image_array[image_array > 255] = 0
            image_array = image_array.astype("float32")
            image_array = np.nan_to_num(image_array, nan=0.0)
            image_array = image_array * (255 / np.max(image_array))
            image_array = image_array.astype("uint8")
        else:
            self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
            return

        # Convert to PIL Image and save
        pil_image = PILImage.fromarray(image_array)
        if msg.encoding == 'mono8' or msg.encoding == '32FC1':
            pil_image = pil_image.convert("L")  # Grayscale
        elif msg.encoding in ['rgb8', 'bgra8']:
            pil_image = pil_image.convert("RGB")  # RGB

        self.curr_img = np.array(pil_image).astype("uint8")


    def info_callback(self, data):
        self.K = np.array(data).reshape(3,3)

    def wait_for_message(self):
        print("waiting for image to be published...")
        while len(self.curr_img) == 0 or len(self.K) == 0:
            continue
        print("publish detected!")


if __name__ == "__main__":

    PORT = "/dev/ttyACM0"
    ENABLE_TORQUE = False

    koch_robot = KochRobot(port=PORT, torque=ENABLE_TORQUE)

    # Enable ROS node and run on background thread
    rclpy.init(args=None)
    cam_node = CameraNode()
    executor = SingleThreadedExecutor()
    executor.add_node(cam_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait until image has been published
    cam_node.wait_for_message()

    do = input("Action (read=r, manual=m, calib=c):")
    if do == "r":
        iters = int(input("Number of read iterations?"))
        koch_robot.read_pos(iters)
    elif do == "m":
        koch_robot.manual_control()
    elif do == "c":
        koch_robot.camera_calibration(cam_node)

    koch_robot.exit()

    
