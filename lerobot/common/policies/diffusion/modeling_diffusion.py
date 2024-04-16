"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on Robomimic for SpatialSoftmax.
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
  - Move EMA out of policy.
  - Consolidate _DiffusionUnetImagePolicy into DiffusionPolicy.
  - One more pass on comments and documentation.
"""

import copy
import logging
import math
import time
from collections import deque
from itertools import chain
from typing import Callable

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.models.base_nets import SpatialSoftmax
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion"

    def __init__(self, cfg: DiffusionConfig | None, lr_scheduler_num_training_steps: int = 0):
        super().__init__()
        """
        Args:
            cfg: Policy configuration class instance or None, in which case the default instantiation of the
                 configuration class is used.
        """
        # TODO(alexander-soare): LR scheduler will be removed.
        assert lr_scheduler_num_training_steps > 0
        if cfg is None:
            cfg = DiffusionConfig()
        self.cfg = cfg

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = _DiffusionUnetImagePolicy(cfg)

        # TODO(alexander-soare): This should probably be managed outside of the policy class.
        self.ema_diffusion = None
        self.ema = None
        if self.cfg.use_ema:
            self.ema_diffusion = copy.deepcopy(self.diffusion)
            self.ema = _EMA(cfg, model=self.ema_diffusion)

        # TODO(alexander-soare): Move optimizer out of policy.
        self.optimizer = torch.optim.Adam(
            self.diffusion.parameters(), cfg.lr, cfg.adam_betas, cfg.adam_eps, cfg.adam_weight_decay
        )

        # TODO(alexander-soare): Move LR scheduler out of policy.
        # TODO(rcadene): modify lr scheduler so that it doesn't depend on epochs but steps
        self.global_step = 0

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=lr_scheduler_num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.cfg.n_obs_steps),
            "observation.state": deque(maxlen=self.cfg.n_obs_steps),
            "action": deque(maxlen=self.cfg.n_action_steps),
        }

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], **_) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.

        Note: this method uses the ema model weights if self.training == False, otherwise the non-ema model
        weights.
        """
        assert "observation.image" in batch
        assert "observation.state" in batch
        assert len(batch) == 2

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}
            if not self.training and self.ema_diffusion is not None:
                actions = self.ema_diffusion.generate_actions(batch)
            else:
                actions = self.diffusion.generate_actions(batch)
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch, **_):
        start_time = time.time()

        self.diffusion.train()

        loss = self.diffusion.compute_loss(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.step(self.diffusion)

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "update_s": time.time() - start_time,
        }

        return info

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        missing_keys, unexpected_keys = self.load_state_dict(d, strict=False)
        if len(missing_keys) > 0:
            assert all(k.startswith("ema_diffusion.") for k in missing_keys)
            logging.warning(
                "DiffusionPolicy.load expected ema parameters in loaded state dict but none were found."
            )
        assert len(unexpected_keys) == 0


class _DiffusionUnetImagePolicy(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        self.rgb_encoder = _RgbEncoder(cfg)
        self.unet = _ConditionalUnet1D(
            cfg, global_cond_dim=(cfg.action_dim + self.rgb_encoder.feature_dim) * cfg.n_obs_steps
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            variance_type="fixed_small",
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )

        if cfg.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = cfg.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.cfg.horizon, self.cfg.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
        }
        """
        assert set(batch).issuperset({"observation.state", "observation.image"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.cfg.n_obs_steps

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

        # run sampling
        sample = self.conditional_sample(batch_size, global_cond=global_cond)

        # `horizon` steps worth of actions (from the first observation).
        actions = sample[..., : self.cfg.action_dim]
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.cfg.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "observation.image", "action", "action_is_pad"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.cfg.horizon
        assert n_obs_steps == self.cfg.n_obs_steps

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

        trajectory = batch["action"]

        # Forward diffusion.
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.cfg.prediction_type == "epsilon":
            target = eps
        elif self.cfg.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.cfg.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class _RgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if all(v == 1.0 for v in chain(cfg.image_normalization_mean, cfg.image_normalization_std)):
            self.normalizer = nn.Identity()
        else:
            self.normalizer = torchvision.transforms.Normalize(
                mean=cfg.image_normalization_mean, std=cfg.image_normalization_std
            )
        if cfg.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(cfg.crop_shape)
            if cfg.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(cfg.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, cfg.vision_backbone)(
            pretrained=cfg.use_pretrained_backbone
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if cfg.use_group_norm:
            if cfg.use_pretrained_backbone:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        with torch.inference_mode():
            feat_map_shape = tuple(self.backbone(torch.zeros(size=(1, 3, *cfg.image_size))).shape[1:])
        self.pool = SpatialSoftmax(feat_map_shape, num_kp=cfg.spatial_softmax_num_keypoints)
        self.feature_dim = cfg.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(cfg.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: normalize and maybe crop (if it was set up in the __init__).
        x = self.normalizer(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class _SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class _ConditionalUnet1D(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, cfg: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.cfg = cfg

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            _SinusoidalPosEmb(cfg.diffusion_step_embed_dim),
            nn.Linear(cfg.diffusion_step_embed_dim, cfg.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(cfg.diffusion_step_embed_dim * 4, cfg.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = cfg.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(cfg.action_dim, cfg.down_dims[0])] + list(
            zip(cfg.down_dims[:-1], cfg.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": cfg.kernel_size,
            "n_groups": cfg.n_groups,
            "use_film_scale_modulation": cfg.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock1D(dim_in, dim_out, **common_res_block_kwargs),
                        _ConditionalResidualBlock1D(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                _ConditionalResidualBlock1D(cfg.down_dims[-1], cfg.down_dims[-1], **common_res_block_kwargs),
                _ConditionalResidualBlock1D(cfg.down_dims[-1], cfg.down_dims[-1], **common_res_block_kwargs),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        _ConditionalResidualBlock1D(dim_in * 2, dim_out, **common_res_block_kwargs),
                        _ConditionalResidualBlock1D(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            _Conv1dBlock(cfg.down_dims[0], cfg.down_dims[0], kernel_size=cfg.kernel_size),
            nn.Conv1d(cfg.down_dims[0], cfg.action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class _ConditionalResidualBlock1D(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = _Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = _Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class _EMA:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, cfg: DiffusionConfig, model: nn.Module):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_alpha (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = cfg.ema_update_after_step
        self.inv_gamma = cfg.ema_inv_gamma
        self.power = cfg.ema_power
        self.min_alpha = cfg.ema_min_alpha
        self.max_alpha = cfg.ema_max_alpha

        self.alpha = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_alpha, min(value, self.max_alpha))

    @torch.no_grad()
    def step(self, new_model):
        self.alpha = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules(), strict=True):
            # Iterate over immediate parameters only.
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=True
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    # Copy BatchNorm parameters, and non-trainable parameters directly.
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.alpha)

        self.optimization_step += 1