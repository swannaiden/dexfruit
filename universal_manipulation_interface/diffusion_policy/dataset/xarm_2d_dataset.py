# # diffusion_policy/dataset/xarm_2d_dataset.py
# from typing import Dict, List, Tuple, Optional
# import time
# import numpy as np
# import torch
# import copy
# import zarr
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.model.common.normalizer import LinearNormalizer
# from diffusion_policy.common.normalize_util import get_image_range_normalizer
# from diffusion_policy.dataset.base_dataset import BaseImageDataset
# import cv2
# import os

# class XArmImageDataset2D(BaseImageDataset):
#     """
#     Dataset for *2-D* Diffusion Policy on your xArm recordings.
#     Dynamically loads observations based on shape_meta configuration.
#     """

#     def __init__(
#         self,
#         zarr_path: str,
#         shape_meta: Dict[str, Dict] = None,
#         horizon: int = 2,
#         pad_before: int = 0,
#         pad_after: int = 0,
#         seed: int = 42,
#         val_ratio: float = 0.0,
#         max_train_episodes: Optional[int] = None,
#     ):
#         super().__init__()
        
#         # Parse shape_meta configuration
#         self.obs_config = {}
#         self.rgb_keys = []
#         self.non_rgb_keys = []
        
#         if shape_meta is not None and 'obs' in shape_meta:
#             obs_meta = shape_meta['obs']
#             for key, meta in obs_meta.items():
#                 self.obs_config[key] = meta
#                 if meta.get('type') == 'rgb':
#                     self.rgb_keys.append(key)
#                 else:
#                     self.non_rgb_keys.append(key)
#         else:
#             # Default configuration if no shape_meta provided
#             self.rgb_keys = ['rs_side_rgb', 'rs_front_rgb']
#             self.non_rgb_keys = ['pose']
#             self.obs_config = {
#                 'rs_side_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
#                 'rs_front_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
#                 'pose': {'shape': [10]}
#             }
        
#         print(f"Loaded observation config:")
#         print(f"  RGB keys: {self.rgb_keys}")
#         print(f"  Non-RGB keys: {self.non_rgb_keys}")
        
#         # Open the Zarr store
#         self.root = zarr.open(zarr_path, mode="r")

#         # List of episodes
#         self.episodes: List[str] = sorted(self.root.group_keys())

#         # Split episodes
#         np.random.seed(seed)
#         n_eps = len(self.episodes)
#         idxs = np.arange(n_eps)
#         np.random.shuffle(idxs)
#         n_val = int(val_ratio * n_eps)
#         val_idxs = set(idxs[:n_val])
#         train_idxs = [i for i in idxs if i not in val_idxs]

#         if max_train_episodes is not None:
#             train_idxs = list(train_idxs)[:max_train_episodes]

#         self.train_eps = {self.episodes[i] for i in train_idxs}
#         self.val_eps = {self.episodes[i] for i in val_idxs}

#         # Build index list: (episode, start_t) for training
#         self.horizon = horizon
#         self.pad_before = pad_before
#         self.pad_after = pad_after
#         self.index: List[Tuple[str,int]] = []
        
#         for epi in self.train_eps:
#             grp = self.root[epi]
#             # Find minimum length across all RGB observations
#             min_T = float('inf')
#             for rgb_key in self.rgb_keys:
#                 if rgb_key in grp:
#                     T = grp[rgb_key].shape[0]
#                     min_T = min(min_T, T)
#                 else:
#                     print(f"Warning: {rgb_key} not found in episode {epi}")
            
#             if min_T > horizon:
#                 for t in range(min_T - horizon):
#                     self.index.append((epi, t))

#     def get_validation_dataset(self) -> "XArmImageDataset2D":
#         # shallow copy, swap to validation episodes
#         val_ds = copy.copy(self)
#         val_ds.index = []
#         for epi in self.val_eps:
#             grp = val_ds.root[epi]
#             # Find minimum length across all RGB observations
#             min_T = float('inf')
#             for rgb_key in val_ds.rgb_keys:
#                 if rgb_key in grp:
#                     T = grp[rgb_key].shape[0]
#                     min_T = min(min_T, T)
            
#             if min_T > val_ds.horizon:
#                 for t in range(min_T - val_ds.horizon):
#                     val_ds.index.append((epi, t))
#         return val_ds

#     def __len__(self) -> int:
#         return len(self.index)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         epi, t0 = self.index[idx]
#         grp = self.root[epi]
#         sl = slice(t0, t0 + self.horizon)

#         obs_dict = {}
        
#         # Load RGB observations
#         for rgb_key in self.rgb_keys:
#             if rgb_key in grp:
#                 rgb_data = grp[rgb_key][sl]  # (H, h, w, 3) or (H, 3, h, w)
                
#                 # If your Zarr stores HWC, move channels to front
#                 if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
#                     # (T, H, W, C) -> (T, C, H, W)
#                     rgb_img = np.moveaxis(rgb_data, -1, 1)
#                 else:
#                     # assume already (T, C, H, W)
#                     rgb_img = rgb_data
                
#                 # Normalize to [0, 1]
#                 rgb_img = rgb_img.astype(np.float32) / 255.0
#                 obs_dict[rgb_key] = rgb_img
        
#         # Load non-RGB observations (e.g., pose)
#         for key in self.non_rgb_keys:
#             if key in grp:
#                 obs_dict[key] = grp[key][sl].astype(np.float32)
        
#         # Load action
#         action = grp["action"][sl].astype(np.float32)
        
#         # # Debug visualization - save images every ~1000 samples
#         # if np.random.randint(0, 1000) == 0:
#         #     save_dir = "debug_images"
#         #     os.makedirs(save_dir, exist_ok=True)
            
#         #     # Save action and non-RGB observations
#         #     np.save(os.path.join(save_dir, "action.npy"), action)
#         #     for key in self.non_rgb_keys:
#         #         if key in obs_dict:
#         #             np.save(os.path.join(save_dir, f"{key}.npy"), obs_dict[key])
            
#         #     # Save the last frame from each RGB camera
#         #     for rgb_key in self.rgb_keys:
#         #         if rgb_key in obs_dict:
#         #             rgb_img_last = obs_dict[rgb_key][-1]
                    
#         #             # Convert from (C, H, W) to (H, W, C) and scale to uint8
#         #             rgb_img_save = (np.transpose(rgb_img_last, (1, 2, 0)) * 255).astype(np.uint8)
                    
#         #             # Convert RGB to BGR for cv2
#         #             rgb_img_save = cv2.cvtColor(rgb_img_save, cv2.COLOR_RGB2BGR)
                    
#         #             # Save the image
#         #             cv2.imwrite(os.path.join(save_dir, f"{rgb_key}.png"), rgb_img_save)
                    
#         #             # print(f"{rgb_key} shape: {obs_dict[rgb_key].shape}")
            
#         #     # print(f"Debug data saved to {save_dir}/")
#         #     # print(f"action shape: {action.shape}")

#         sample = {
#             "obs": obs_dict,
#             "action": action,
#         }
#         return dict_apply(sample, torch.from_numpy)

#     def get_normalizer(self, mode: str = "limits", **kwargs):
#         # Gather data for normalization
#         data_dict = {}
        
#         # Collect non-RGB observations
#         for key in self.non_rgb_keys:
#             data_list = []
#             for epi in self.train_eps:
#                 grp = self.root[epi]
#                 if key in grp:
#                     data_list.append(grp[key][...])
#             if data_list:
#                 data_dict[key] = np.concatenate(data_list, axis=0)
        
#         # Collect actions
#         acts = []
#         for epi in self.train_eps:
#             grp = self.root[epi]
#             acts.append(grp["action"][...])
#         data_dict["action"] = np.concatenate(acts, axis=0)
        
#         # Create normalizer
#         normalizer = LinearNormalizer()
#         normalizer.fit(data=data_dict, last_n_dims=1, mode=mode, **kwargs)
        
#         # Add identity normalizers for RGB images
#         for rgb_key in self.rgb_keys:
#             normalizer[rgb_key] = get_image_range_normalizer()
        
#         return normalizer


# diffusion_policy/dataset/xarm_2d_dataset.py
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import cv2
import os
import psutil
from tqdm import tqdm


class XArmImageDataset2D(BaseImageDataset):
    """
    Dataset for *2-D* Diffusion Policy on your xArm recordings.
    Dynamically loads observations based on shape_meta configuration.
    Now supports full RAM preloading for maximum speed.
    """

    def __init__(
        self,
        zarr_path: str,
        shape_meta: Dict[str, Dict] = None,
        horizon: int = 2,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        preload: bool = True,  # NEW: Enable full RAM preloading
        # Vision blackout ablation parameters
        vision_blackout_enabled: bool = False,
        vision_blackout_gripper_threshold: float = 0.1,
        vision_blackout_keys: Optional[List[str]] = None,
        # Tactile-based vision blackout parameters
        vision_blackout_use_tactile: bool = False,
        vision_blackout_tactile_metric: str = 'energy',  # 'energy', 'mean', 'std', 'max', 'percentile_95'
        vision_blackout_tactile_threshold: float = 0.1,
        vision_blackout_tactile_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.is_training = True  # Track if we're in training mode
        
        self.preload = preload
        self.preloaded_data = {}  # Will store all episodes if preloading
        self.shape_meta = shape_meta  # Store shape_meta for horizon access
        
        # Vision blackout ablation settings
        self.vision_blackout_enabled = vision_blackout_enabled
        self.vision_blackout_gripper_threshold = vision_blackout_gripper_threshold
        self.vision_blackout_keys = vision_blackout_keys or ['rs_wrist_rgb']  # Default to camera keys
        
        # Tactile-based vision blackout settings
        self.vision_blackout_use_tactile = vision_blackout_use_tactile
        self.vision_blackout_tactile_metric = vision_blackout_tactile_metric
        self.vision_blackout_tactile_threshold = vision_blackout_tactile_threshold
        self.vision_blackout_tactile_keys = vision_blackout_tactile_keys or ['dt_left_diff', 'dt_right_diff']
        
        # Parse shape_meta configuration
        self.rgb_keys = []
        self.non_rgb_keys = []
        self.obs_config = {}
        
        if shape_meta is not None and 'obs' in shape_meta:
            obs_meta = shape_meta['obs']
            for key, meta in obs_meta.items():
                self.obs_config[key] = meta
                if meta.get('type') == 'rgb':
                    self.rgb_keys.append(key)
                else:
                    self.non_rgb_keys.append(key)
            
            # Also store action config if present
            if 'action' in shape_meta:
                self.obs_config['action'] = shape_meta['action']
        else:
            # Default configuration if no shape_meta provided
            self.rgb_keys = ['rs_side_rgb', 'rs_front_rgb']
            self.non_rgb_keys = ['pose']
            self.obs_config = {
                'rs_side_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
                'rs_front_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
                'pose': {'shape': [10]}
            }
        
        print(f"Loaded observation config:")
        print(f"  RGB keys: {self.rgb_keys}")
        print(f"  Non-RGB keys: {self.non_rgb_keys}")
        
        # Print horizon information for debugging
        for key in self.rgb_keys:
            obs_horizon = self.obs_config[key].get('horizon', horizon)
            print(f"  {key} horizon: {obs_horizon}")
        for key in self.non_rgb_keys:
            obs_horizon = self.obs_config[key].get('horizon', horizon)
            print(f"  {key} horizon: {obs_horizon}")
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', horizon)
            print(f"  action horizon: {action_horizon}")
        
        # Open the Zarr store
        self.root = zarr.open(zarr_path, mode="r")

        # List of episodes
        self.episodes: List[str] = sorted(self.root.group_keys())

        # Split episodes
        np.random.seed(seed)
        n_eps = len(self.episodes)
        idxs = np.arange(n_eps)
        np.random.shuffle(idxs)
        n_val = int(val_ratio * n_eps)
        val_idxs = set(idxs[:n_val])
        train_idxs = [i for i in idxs if i not in val_idxs]

        if max_train_episodes is not None:
            train_idxs = list(train_idxs)[:max_train_episodes]

        self.train_eps = {self.episodes[i] for i in train_idxs}
        self.val_eps = {self.episodes[i] for i in val_idxs}

        # Build index list: (episode, start_t) for training
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.index: List[Tuple[str,int]] = []
        
        # Preload data if requested
        if self.preload:
            print("🚀 Preloading dataset into RAM...")
            self._preload_all_data()
        
        # Build index after preloading
        # Use action horizon as the minimum required timesteps for indexing
        action_horizon = horizon  # default fallback
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', horizon)
        elif self.shape_meta and 'action' in self.shape_meta:
            action_horizon = self.shape_meta['action'].get('horizon', horizon)
        
        print(f"Using action_horizon={action_horizon} for indexing")
        
        for epi in self.train_eps:
            if self.preload:
                min_T = self._get_min_timesteps_preloaded(epi)
            else:
                grp = self.root[epi]
                min_T = self._get_min_timesteps_zarr(grp)
            
            if min_T > action_horizon:  # Use action_horizon instead of horizon
                for t in range(min_T - action_horizon):
                    self.index.append((epi, t))

    def _get_min_timesteps_zarr(self, grp):
        """Get minimum timesteps from zarr group"""
        min_T = float('inf')
        for rgb_key in self.rgb_keys:
            if rgb_key in grp:
                T = grp[rgb_key].shape[0]
                min_T = min(min_T, T)
            else:
                print(f"Warning: {rgb_key} not found in episode")
        return min_T

    def _get_min_timesteps_preloaded(self, epi):
        """Get minimum timesteps from preloaded data"""
        if epi not in self.preloaded_data:
            return 0
        min_T = float('inf')
        for rgb_key in self.rgb_keys:
            if rgb_key in self.preloaded_data[epi]:
                T = self.preloaded_data[epi][rgb_key].shape[0]
                min_T = min(min_T, T)
        return min_T

    def _preload_all_data(self):
        """Load all episodes into RAM"""
        start_time = time.time()
        total_memory_mb = 0
        
        episodes_to_load = self.train_eps | self.val_eps  # Union of train and val
        
        for i, epi in enumerate(tqdm(episodes_to_load, desc="Preloading episodes")):
            # tqdm handles the progress bar, print only the episode name if desired
            # print(f"Loading episode {i+1}/{len(episodes_to_load)}: {epi}")
            grp = self.root[epi]
            
            episode_data = {}
            
            # Load RGB observations
            for rgb_key in self.rgb_keys:
                if rgb_key in grp:
                    rgb_data = grp[rgb_key][...]  # Load entire array
                    
                    # Convert to (T, C, H, W) format and normalize
                    if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
                        rgb_data = np.moveaxis(rgb_data, -1, 1)
                    
                    # Convert to float32 and normalize to [0, 1]
                    rgb_data = rgb_data.astype(np.float32) / 255.0
                    episode_data[rgb_key] = rgb_data
                    
                    # Track memory usage
                    memory_mb = rgb_data.nbytes / (1024 * 1024)
                    total_memory_mb += memory_mb
                    # print(f"  {rgb_key}: {rgb_data.shape} ({memory_mb:.1f} MB)")
            
            # Load non-RGB observations
            for key in self.non_rgb_keys:
                if key in grp:
                    data = grp[key][...].astype(np.float32)
                    episode_data[key] = data
                    memory_mb = data.nbytes / (1024 * 1024)
                    total_memory_mb += memory_mb
                    # print(f"  {key}: {data.shape} ({memory_mb:.1f} MB)")
            
            # Load actions
            action_data = grp["action"][...].astype(np.float32)
            episode_data["action"] = action_data
            memory_mb = action_data.nbytes / (1024 * 1024)
            total_memory_mb += memory_mb
            
            self.preloaded_data[epi] = episode_data
        
        load_time = time.time() - start_time
        print(f"✅ Preloading complete!")
        print(f"   Total data: {total_memory_mb/1024:.2f} GB")
        print(f"   Load time: {load_time:.1f} seconds")
        print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    def get_validation_dataset(self) -> "XArmImageDataset2D":
        """Create validation dataset - shares preloaded data if available"""
        val_ds = copy.copy(self)
        val_ds.index = []
        val_ds.is_training = False  # Disable noise augmentation for validation
        
        # Use action horizon for indexing (same as training)
        action_horizon = self.horizon  # default fallback
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', self.horizon)
        elif self.shape_meta and 'action' in self.shape_meta:
            action_horizon = self.shape_meta['action'].get('horizon', self.horizon)
        
        for epi in self.val_eps:
            if self.preload:
                min_T = self._get_min_timesteps_preloaded(epi)
            else:
                grp = val_ds.root[epi]
                min_T = self._get_min_timesteps_zarr(grp)
            
            if min_T > action_horizon:  # Use action_horizon instead of val_ds.horizon
                for t in range(min_T - action_horizon):
                    val_ds.index.append((epi, t))
        return val_ds

    def __len__(self) -> int:
        return len(self.index)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, t0 = self.index[idx]

        if self.preload:
            # Fast path: use preloaded data
            episode_data = self.preloaded_data[epi]
            
            obs_dict = {}
            for rgb_key in self.rgb_keys:
                if rgb_key in episode_data:
                    # Use the horizon from shape_meta for this observation
                    obs_horizon = self.obs_config[rgb_key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[rgb_key] = episode_data[rgb_key][sl]
            
            for key in self.non_rgb_keys:
                if key in episode_data:
                    # Use the horizon from shape_meta for this observation
                    obs_horizon = self.obs_config[key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[key] = episode_data[key][sl]
            
            # Action always uses the full horizon (action_horizon from shape_meta)
            action_horizon = self.obs_config.get('action', {}).get('horizon', self.horizon)
            if 'action' in self.obs_config:
                action_horizon = self.obs_config['action'].get('horizon', self.horizon)
            else:
                # Fallback: look in shape_meta for action horizon
                action_horizon = self.shape_meta.get('action', {}).get('horizon', self.horizon)
            sl_action = slice(t0, t0 + action_horizon)
            action = episode_data["action"][sl_action]
            
        else:
            # Slow path: load from zarr (fallback)
            grp = self.root[epi]
            obs_dict = {}
            
            # Load RGB observations with individual horizons
            for rgb_key in self.rgb_keys:
                if rgb_key in grp:
                    obs_horizon = self.obs_config[rgb_key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    rgb_data = grp[rgb_key][sl]
                    
                    if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
                        rgb_img = np.moveaxis(rgb_data, -1, 1)
                    else:
                        rgb_img = rgb_data
                    
                    rgb_img = rgb_img.astype(np.float32) / 255.0
                    obs_dict[rgb_key] = rgb_img
            
            # Load non-RGB observations with individual horizons
            for key in self.non_rgb_keys:
                if key in grp:
                    obs_horizon = self.obs_config[key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[key] = grp[key][sl].astype(np.float32)
            
            # Load action with action horizon
            action_horizon = self.obs_config.get('action', {}).get('horizon', self.horizon)
            if 'action' in self.obs_config:
                action_horizon = self.obs_config['action'].get('horizon', self.horizon)
            else:
                # Fallback: look in shape_meta for action horizon
                action_horizon = self.shape_meta.get('action', {}).get('horizon', self.horizon)
            sl_action = slice(t0, t0 + action_horizon)
            action = grp["action"][sl_action].astype(np.float32)

        # Apply noise augmentation to pose data during training
        # obs_dict = self._apply_pose_noise(obs_dict)

        # Apply vision blackout ablation if enabled
        if self.vision_blackout_enabled:
            obs_dict = self._apply_vision_blackout(obs_dict)

        sample = {
            "obs": obs_dict,
            "action": action,
        }
        return dict_apply(sample, torch.from_numpy)

    def _compute_tactile_metric(self, tactile_data: np.ndarray, metric: str) -> float:
        """Compute a specific metric for tactile data.
        
        Args:
            tactile_data: Tactile data array, shape (C, H, W)
            metric: Metric to compute ('energy', 'mean', 'std', 'max', 'percentile_95')
            
        Returns:
            Computed metric value
        """
        # Convert to grayscale if needed (take mean across channels)
        if tactile_data.shape[0] == 3:  # RGB
            gray = np.mean(tactile_data, axis=0)
        else:
            gray = tactile_data[0]  # Take first channel
            
        if metric == 'energy':
            return np.sum(gray ** 2)
        elif metric == 'mean':
            return np.mean(gray)
        elif metric == 'std':
            return np.std(gray)
        elif metric == 'max':
            return np.max(gray)
        elif metric == 'percentile_95':
            return np.percentile(gray, 95)
        else:
            raise ValueError(f"Unknown tactile metric: {metric}")

    def _apply_vision_blackout(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply vision blackout based on gripper value or tactile activity.
        
        Args:
            obs_dict: Dictionary of observations
            
        Returns:
            Modified obs_dict with vision keys blacked out when conditions are met
        """
        if self.vision_blackout_use_tactile:
            # Use tactile-based blackout
            blackout_mask = self._get_tactile_blackout_mask(obs_dict)
        else:
            # Use gripper-based blackout (original method)
            blackout_mask = self._get_gripper_blackout_mask(obs_dict)
        
        if blackout_mask is None or not np.any(blackout_mask):
            # No blackout needed, return original
            return obs_dict
        
        # Only copy the dict if we actually need to modify something
        obs_dict = obs_dict.copy()
        
        # Apply blackout to specified vision keys
        for key in self.vision_blackout_keys:
            if key in obs_dict and key in self.rgb_keys:
                # Only copy the data if we need to modify it
                obs_data = obs_dict[key]
                needs_modification = False
                
                # Check if any timesteps need blackout
                if obs_data.ndim == 3:
                    # Single timestep RGB: (C, H, W)
                    needs_modification = len(blackout_mask) == 1 and blackout_mask[0]
                elif obs_data.ndim == 4:
                    # Multiple timesteps RGB: (T, C, H, W)
                    needs_modification = np.any(blackout_mask[:min(len(blackout_mask), obs_data.shape[0])])
                
                if needs_modification:
                    # Only now create a copy
                    obs_data = obs_data.copy()
                    
                    # Handle different observation shapes
                    if obs_data.ndim == 3:
                        # Single timestep RGB: (C, H, W)
                        if len(blackout_mask) == 1 and blackout_mask[0]:
                            obs_data[:] = 0.0  # Black out all channels
                    elif obs_data.ndim == 4:
                        # Multiple timesteps RGB: (T, C, H, W)
                        for t in range(min(len(blackout_mask), obs_data.shape[0])):
                            if blackout_mask[t]:
                                obs_data[t, :] = 0.0  # Black out all channels for this timestep
                    
                    obs_dict[key] = obs_data
                
        return obs_dict
    
    def _get_gripper_blackout_mask(self, obs_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Get blackout mask based on gripper values."""
        # Check if pose data is available
        if 'pose' not in obs_dict:
            return None
            
        # Get pose data - shape should be (horizon, 10) where gripper is at index 9
        pose_data = obs_dict['pose']
        
        # Handle different pose shapes
        if pose_data.ndim == 1:
            # Single timestep
            gripper_values = pose_data[9:10]  # Keep as array for consistent indexing
        elif pose_data.ndim == 2:
            # Multiple timesteps (horizon, 10)
            gripper_values = pose_data[:, 9]  # Shape: (horizon,)
        else:
            print(f"Warning: Unexpected pose shape {pose_data.shape}, skipping vision blackout")
            return None
        
        # Check which timesteps should have vision blacked out
        return gripper_values >= self.vision_blackout_gripper_threshold
    
    def _get_tactile_blackout_mask(self, obs_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Get blackout mask based on tactile activity."""
        # Check if any tactile data is available
        available_tactile_keys = [key for key in self.vision_blackout_tactile_keys if key in obs_dict]
        if not available_tactile_keys:
            print("Warning: No tactile data available for tactile-based blackout")
            return None
        
        # Determine horizon from first available tactile key
        first_tactile_data = obs_dict[available_tactile_keys[0]]
        if first_tactile_data.ndim == 3:
            # Single timestep (C, H, W)
            horizon = 1
        elif first_tactile_data.ndim == 4:
            # Multiple timesteps (T, C, H, W)
            horizon = first_tactile_data.shape[0]
        else:
            print(f"Warning: Unexpected tactile data shape {first_tactile_data.shape}")
            return None
        
        blackout_mask = np.zeros(horizon, dtype=bool)
        
        # Check each timestep
        for t in range(horizon):
            tactile_activity_detected = False
            
            # Check activity across all available tactile sensors
            for key in available_tactile_keys:
                tactile_data = obs_dict[key]
                
                # Extract data for current timestep
                if tactile_data.ndim == 3:
                    # Single timestep
                    current_tactile = tactile_data
                else:
                    # Multiple timesteps
                    current_tactile = tactile_data[t]
                
                # Compute the specified metric
                metric_value = self._compute_tactile_metric(current_tactile, self.vision_blackout_tactile_metric)
                
                # Check if above threshold
                if metric_value >= self.vision_blackout_tactile_threshold:
                    tactile_activity_detected = True
                    break
            
            blackout_mask[t] = tactile_activity_detected
        
        return blackout_mask

    def get_normalizer(self, mode: str = "limits", **kwargs):
        # Gather data for normalization
        data_dict = {}
        
        if self.preload:
            # Fast path: use preloaded data
            for key in self.non_rgb_keys:
                data_list = []
                for epi in self.train_eps:
                    if epi in self.preloaded_data and key in self.preloaded_data[epi]:
                        data_list.append(self.preloaded_data[epi][key])
                if data_list:
                    data_dict[key] = np.concatenate(data_list, axis=0)
            
            # Collect actions
            acts = []
            for epi in self.train_eps:
                if epi in self.preloaded_data:
                    acts.append(self.preloaded_data[epi]["action"])
            data_dict["action"] = np.concatenate(acts, axis=0)
            
        else:
            # Slow path: load from zarr
            for key in self.non_rgb_keys:
                data_list = []
                for epi in self.train_eps:
                    grp = self.root[epi]
                    if key in grp:
                        data_list.append(grp[key][...])
                if data_list:
                    data_dict[key] = np.concatenate(data_list, axis=0)
            
            acts = []
            for epi in self.train_eps:
                grp = self.root[epi]
                acts.append(grp["action"][...])
            data_dict["action"] = np.concatenate(acts, axis=0)
        
        # Create normalizer
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dict, last_n_dims=1, mode=mode, **kwargs)
        
        # Add identity normalizers for RGB images
        for rgb_key in self.rgb_keys:
            normalizer[rgb_key] = get_image_range_normalizer()
        
        return normalizer