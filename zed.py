import pyzed.sl as sl
import numpy as np
import cv2
from scipy.ndimage import label
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import matplotlib.pyplot as plt


class ZEDCamera: 

    def __init__(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_maximum_distance = 40 

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


    def pixel_to_3d_points(self):

        try:
            x = np.load("camera_extrinsics.npy", allow_pickle=True)
            R, t = x[0].reshape(3,3), x[1].reshape(3,1)
            t = t / 100
            print("R:", R, "\nt:", t)
        except:
            print("Failed to load extrinsics.")
            R, t = np.eye(3), np.array([[0], [0], [0]])

        depth_pc = self.capture_points()
        K_inv = np.linalg.inv(self.get_K())
        R_inv = np.linalg.inv(R)

        # Get array of valid pixel locations
        xv, yv = np.meshgrid(np.arange(depth_pc.shape[1]), np.arange(depth_pc.shape[0]))
        nan_mask = ~np.isnan(depth_pc)
        xv, yv = xv[nan_mask], yv[nan_mask]
        pc_all = np.vstack((xv, yv, np.ones(xv.shape)))
        
        # Convert pixel to world coordinates
        s = depth_pc[yv, xv]
        pc_camera = s * (K_inv @ pc_all)
        pw_final = (R_inv @ (pc_camera - t)).T

        return pw_final, pc_all, R, t


    def sam2_masks(self):
        image = self.capture_image("rgb")
        mask_dict = self.mask_generator.generate(image)
        masks = [mask_dict[i]["segmentation"] for i in range(len(mask_dict))]
        return masks