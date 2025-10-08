from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.model.vision.tactile_obs_encoder import TactileObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetDualTimmPolicy(BaseImagePolicy):
    """Diffusion policy using separate encoders for RGB cameras and
    DenseTact tactile images."""

    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 image_obs_encoder: TimmObsEncoder,
                 tactile_obs_encoder: TactileObsEncoder,
                 num_inference_steps=None,
                 obs_as_global_cond: bool = True,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=(256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True,
                 input_pertub: float = 0.1,
                 inpaint_fixed_action_prefix: bool = False,
                 train_diffusion_n_samples: int = 1,
                 **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']

        obs_feature_dim = (
            np.prod(image_obs_encoder.output_shape()) +
            np.prod(tactile_obs_encoder.output_shape())
        )

        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.image_obs_encoder = image_obs_encoder
        self.tactile_obs_encoder = tactile_obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t,
                                 local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]):
        cam_keys = self.image_obs_encoder.rgb_keys + self.image_obs_encoder.low_dim_keys
        cam_obs = {k: nobs[k] for k in cam_keys if k in nobs}
        cam_feat = self.image_obs_encoder(cam_obs)

        tactile_keys = self.tactile_obs_encoder.rgb_keys + self.tactile_obs_encoder.low_dim_keys
        tactile_obs = {k: nobs[k] for k in tactile_keys if k in nobs}
        tactile_feat = self.tactile_obs_encoder(tactile_obs)

        return torch.cat([cam_feat, tactile_feat], dim=-1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       fixed_action_prefix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        global_cond = self._encode_obs(nobs)

        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)

        action_pred = self.normalizer['action'].unnormalize(nsample)
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        global_cond = self._encode_obs(nobs)

        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(
                global_cond, repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],), device=trajectory.device).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
