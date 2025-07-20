import copy
from typing import List, Optional, Union

import lightning as pl
import torch
import torch.nn as nn
from ...utils import instantiate_from_config


class ConsistencyModel(pl.LightningModule):
    """Consistency model using a transformer backbone."""

    def __init__(
        self,
        network: Union[dict, nn.Module],
        dimension: int,
        data_size: List[int],
        sim_fields: List[str],
        sim_params: List[str],
        conditioning_length: int,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        ema_rate: float = 0.9999,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps: int = 40,
        num_steps_eval: int = 1,
        rand_cons_step: bool = False,
        unroll_steps: int = 1,
        target_loss_weight: float = 0.1,
        monitor: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(network, dict):
            self.model: nn.Module = instantiate_from_config(network)
        else:
            self.model = network

        self.save_hyperparameters(ignore=["network"])

        self.dimension = dimension
        self.data_size = data_size
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.conditioning_length = conditioning_length

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.ema_rate = ema_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.num_steps_eval = num_steps_eval
        self.rand_cons_step = rand_cons_step
        self.unroll_steps = unroll_steps
        self.target_loss_weight = target_loss_weight
        field_channels = 0
        for field in sim_fields:
            if field == "vel":
                field_channels += dimension
            else:
                field_channels += 1
        self.target_channels = field_channels + len(sim_params)
        cond_frames = conditioning_length
        channels_per_frame = field_channels + len(sim_params)
        self.cond_channels = cond_frames * channels_per_frame
        self.total_channels = self.cond_channels + self.target_channels

        self.target_model = self._create_target_model()
        self._initialize_target_model()

        if monitor is not None:
            self.monitor = monitor

    def _create_target_model(self) -> nn.Module:
        target_model = copy.deepcopy(self.model)
        for p in target_model.parameters():
            p.requires_grad = False
        return target_model

    def _initialize_target_model(self) -> None:
        with torch.no_grad():
            for t_p, o_p in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                t_p.data.copy_(o_p.data)

    def _update_target_model(self) -> None:
        with torch.no_grad():
            for t_p, o_p in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                t_p.data.mul_(self.ema_rate).add_(o_p.data, alpha=1 - self.ema_rate)

    def get_input(
        self, batch, batch_dim: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data: torch.Tensor = batch["data"]

        if batch_dim:
            cond = data[:, : self.conditioning_length]
            target = data[
                :,
                self.conditioning_length : self.conditioning_length + self.unroll_steps,
            ]
        else:
            cond = data[: self.conditioning_length]
            target = data[
                self.conditioning_length : self.conditioning_length + self.unroll_steps
            ]
            cond = cond.unsqueeze(0)
            target = target.unsqueeze(0)

        return cond, target

    def get_noise_schedule(self, num_steps: int) -> torch.Tensor:
        step_indices = torch.arange(num_steps, dtype=torch.float32)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        t = (
            sigma_max_rho
            + step_indices / (num_steps - 1) * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        t = torch.clamp(t, min=self.sigma_min, max=self.sigma_max)
        return t

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_c = torch.clamp(sigma, min=self.sigma_min)
        return 0.25 * torch.log(sigma_c)

    def consistency_function(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        use_target: bool = False,
        return_cond: bool = False,
    ) -> torch.Tensor:
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0])
        elif sigma.dim() == 1 and sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0])

        sigma_r = sigma.view(-1, 1, 1, 1)
        boundary_mask = (sigma <= self.sigma_min).float().view(-1, 1, 1, 1)
        process_mask = (sigma > self.sigma_min).float().view(-1, 1, 1, 1)
        output = x.clone()

        if process_mask.sum() > 0:
            c_skip_val = self.c_skip(sigma_r)
            c_out_val = self.c_out(sigma_r)
            c_in_val = self.c_in(sigma_r)
            c_noise_val = self.c_noise(sigma).view(-1)

            x_in = torch.cat([cond, c_in_val * x], dim=1)
            model = self.target_model if use_target else self.model
            f_theta_full = model(x_in, c_noise_val).sample
            f_theta_cond = f_theta_full[:, : cond.shape[1], :, :]
            f_theta = f_theta_full[:, cond.shape[1] :, :, :]
            consistency_out = c_skip_val * x + c_out_val * f_theta
            output = boundary_mask * x + process_mask * consistency_out

        if return_cond:
            return output, f_theta_cond
        return output

    def consistency_loss(
        self, target_seq: torch.Tensor, cond_seq: torch.Tensor
    ) -> torch.Tensor:
        assert self.unroll_steps == target_seq.shape[1]
        device = target_seq.device
        B = target_seq.size(0)
        C_h = cond_seq.size(2)
        T = self.num_steps
        U = self.unroll_steps
        t_schedule = self.get_noise_schedule(T).to(device)
        total_loss = 0.0
        cond = cond_seq.view(B, cond_seq.size(1) * C_h, *cond_seq.shape[3:])
        
        for i in range(U):
            target = target_seq[:, i]
            if self.rand_cons_step:
                t_idx = torch.randint(0, T - 1, (B,), device=device)
                range_len = (T - 1) - t_idx
                u = torch.rand(B, device=device)
                offset = (u * range_len.float()).floor().long()
                s_idx = t_idx + 1 + offset
            else:
                t_idx = torch.randint(0, T - 1, (B,), device=device)
                s_idx = t_idx + 1
            sigma_t = t_schedule[t_idx]
            sigma_s = t_schedule[s_idx]
            noise = torch.randn_like(target)
            x_t = target + sigma_t.view(-1, 1, 1, 1) * noise
            x_s = target + sigma_s.view(-1, 1, 1, 1) * noise
            with torch.no_grad():
                f_s, _ = self.consistency_function(
                    x_s, sigma_s, cond, use_target=True, return_cond=True
                )
            f_t, cond_t = self.consistency_function(
                x_t, sigma_t, cond, use_target=False, return_cond=True
            )
            loss_cons = torch.nn.functional.mse_loss(f_t, f_s)
            loss_cond = torch.nn.functional.mse_loss(cond_t, cond)
            target_loss = (
                torch.nn.functional.mse_loss(f_t, target) * self.target_loss_weight
            )
            total_loss += loss_cons + loss_cond + target_loss
            if cond_seq.shape[1] > 1:
                cond = torch.cat([cond[:, C_h:], f_t], dim=1)
            else:
                cond = f_t
        return total_loss / U

    def generate_samples(
        self, cond_seq: torch.Tensor, num_steps: int = 1, use_ema: bool = True
    ) -> torch.Tensor:
        if self.dimension == 3:
            raise NotImplementedError("3D consistency model not implemented")
        device = cond_seq.device
        B = cond_seq.shape[0]
        cond_seq_len = cond_seq.shape[1]
        cond = cond_seq.view(
            B, cond_seq_len * cond_seq.shape[2], cond_seq.shape[3], cond_seq.shape[4]
        )
        model_use_target = use_ema
        if num_steps == 1:
            x = torch.randn(
                B,
                self.target_channels,
                cond_seq.shape[3],
                cond_seq.shape[4],
                device=device,
            )
            sigma = torch.full((B,), self.sigma_max, device=device)
            generated = self.consistency_function(
                x, sigma, cond, use_target=model_use_target
            )
        else:
            t_schedule = self.get_noise_schedule(num_steps).to(device)
            sigma = t_schedule[0]
            x = (
                torch.randn(
                    B,
                    self.target_channels,
                    cond_seq.shape[3],
                    cond_seq.shape[4],
                    device=device,
                )
                * sigma
            )
            for i in range(num_steps - 1):
                sigma = t_schedule[i]
                x0 = self.consistency_function(
                    x, sigma, cond, use_target=model_use_target
                )
                next_sigma = t_schedule[i + 1]
                next_sigma = torch.clamp(
                    next_sigma, min=self.sigma_min, max=self.sigma_max
                )
                noise_scale = torch.sqrt(
                    torch.clamp(next_sigma**2 - self.sigma_min**2, min=0.0)
                )
                noise = torch.randn_like(x)
                x = x0 + noise_scale * noise
            generated = x
        return generated.unsqueeze(1)

    def prediction_step(self, cond: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target is None:
            return self.generate_samples(cond)
        return self.consistency_loss(target, cond)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        cond, target = self.get_input(batch)
        loss = self.forward(cond, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_target_model()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        cond, target = self.get_input(batch)
        val_loss = self.forward(cond, target)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
