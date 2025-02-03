import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from .utils import filter_points_by_bounds
from sklearn.cluster import MeanShift
from rich import print as rprint
from sklearn.cluster import DBSCAN

import pdb

def check_nan(array, name="Array"):
    """
    Check for NaN values in a numpy array or PyTorch tensor.
    
    Args:
    array (np.ndarray or torch.Tensor): The array to check
    name (str): A name for the array (for reporting purposes)
    
    Returns:
    bool: True if NaNs are present, False otherwise
    """
    if isinstance(array, np.ndarray):
        has_nans = np.isnan(array).any()
        nan_count = np.isnan(array).sum()
        inf_count = np.isinf(array).sum()
    elif isinstance(array, torch.Tensor):
        has_nans = torch.isnan(array).any().item()
        nan_count = torch.isnan(array).sum().item()
        inf_count = torch.isinf(array).sum().item()
    else:
        raise TypeError(f"Unsupported type: {type(array)}")
    
    if has_nans:
        rprint(f"[red]{name} contains {nan_count} NaN values and {inf_count} infinity values.[/red]")
        if isinstance(array, np.ndarray):
            rprint(f"[yellow]Array shape: {array.shape}, dtype: {array.dtype}[/yellow]")
        else:
            rprint(f"[yellow]Tensor shape: {array.shape}, dtype: {array.dtype}, device: {array.device}[/yellow]")
        return True
    else:
        rprint(f"[green]{name} does not contain any NaN values.[/green]")
        return False



class KeypointProposer:
    def __init__(self, config):
        self.config = config
        # self.device = torch.device(self.config['device'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rprint(f"[blue]KeypointProposer Using device: {self.device}[/blue]")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], max_iter=400, bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)
        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat, masks, rgb=rgb)
        
        # if 1:
        # # exclude keypoints that are outside of the workspace
        #     # within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=False)
        #     # pdb.set_trace()
        #     filtered_keypoints = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=False)
        #     # Create a boolean mask for the filtered keypoints
        #     within_space = np.array([np.any(np.all(candidate_keypoints[i] == filtered_keypoints, axis=1)) for i in range(len(candidate_keypoints))])
    
        #     candidate_keypoints = candidate_keypoints[within_space]
        #     candidate_pixels = candidate_pixels[within_space]
        #     candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]

        # merge close points by clustering in cartesian space
        merged_indices = self._merge_clusters(candidate_keypoints)

        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]
        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat)
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        # input masks should be binary masks
        # if masks.dtype != bool:
        #     masks = masks.astype(bool)
        
        # Convert NaN values to 0 in points array
        points = np.nan_to_num(points, nan=0.0)
        
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        # shape info
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info
    
    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        projected = rgb.copy()
        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']

        rprint(f"[cyan]Debug: shape_info: {shape_info}[/cyan]")
        rprint(f"[cyan]Debug: transformed_rgb shape: {transformed_rgb.shape}[/cyan]")

        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        
        features_dict = self.dinov2.forward_features(img_tensors) # dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim] 
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
        rprint(f"[cyan]Debug: interpolated_feature_grid shape: {interpolated_feature_grid.shape}[/cyan]")
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim]
        rprint(f"[cyan]Debug: features_flat shape: {features_flat.shape}[/cyan]")
        return features_flat

    def _cluster_features(self, points, features_flat, masks, rgb):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        # pdb.set_trace()
        if len(masks) > 0:
            rprint(f"[cyan]Debug: shape of first mask: {masks[0].shape}[/cyan]")
            
        for rigid_group_id, binary_mask in enumerate(masks):   
            # for mask_id in range(masks_group.shape[0]): # bug: masks_group is a single mask, not a list of masks
            # ignore mask that is too large
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            # TODO 2d cluster or 3d cluster?
            # reshape?
            points = points.reshape(-1,3)
            feature_points = points[binary_mask.reshape(-1)]
        
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch  = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # cluster features to get meaningful regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']): # 5
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]


                # pdb.set_trace()
                member_features = features_pca[member_idx]
                cluster_center = cluster_centers[cluster_id][:3]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                # Skip clusters with no members or all zero coordinates
                if len(member_points[closest_idx]) == 0 or np.all(member_points[closest_idx] == 0):
                    continue
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        rprint(f"[cyan]Debug: Number of clusters: {self.config['num_candidates_per_mask']}[/cyan]")
        rprint(f"[cyan]Debug: Number of candidate keypoints: {len(candidate_keypoints)}[/cyan]")
        # pdb.set_trace()
        
        # # VISUALIZE KEYPOINTS
        # # print(candidate_keypoints)
        # pts = candidate_keypoints.copy()
        # index = 0
        # image = rgb.copy()
        # x = np.load("camera_extrinsics.npy", allow_pickle=True)
        # R, t = x[0].reshape(3,3), x[1].reshape(3,1)
        # t /= 100 # Convert from cm to m
        # K = [[684.46374512, 0, 655.36328125],\
        #      [0, 684.46374512, 357.3269043],\
        #      [0, 0, 1]]
        # from skimage.draw import disk
        # for index in range(len(pts)):
        #     T = np.hstack((R, t))
        #     P = np.array([[pts[index][0], pts[index][1], pts[index][2], 1]])
        #     X = T @ P.T
        #     x_star = K @ X
        #     x_star = x_star / x_star[2]
        #     try:
        #         x, y = int(x_star[0]), int(x_star[1])
        #     except OverflowError:
        #         print(pts)
        #         assert False
        #     rr, cc = disk((y,x), 10, shape=image.shape[:2])
        #     image[rr,cc] = (255, 0, 0)

        # from PIL import Image
        # Image.fromarray(image).convert("RGB").save("candidate_keypoints.png")

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices