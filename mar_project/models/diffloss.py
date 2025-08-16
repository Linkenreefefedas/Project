import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion
#Diffusion Loss 實作，MAR 訓練時的關鍵 loss function 
'''
self.net = SimpleMLPAdaLN
一個簡化的 MLP 網路，帶有 AdaLN（Adaptive LayerNorm modulation），可以根據條件（例如 AR 輸出的上下文特徵）調整特徵分佈 

self.train_diffusion & self.gen_diffusion

訓練：用完整的 diffusion process（timestep_respacing=""）

生成：用指定步數的加速取樣（例如 num_sampling_steps=100）

'''
# 生成 diffusino loss、給去噪後的token

# new start
def _mean_flat(x: torch.Tensor):
    return x.view(x.size(0), -1).mean(dim=1)
#new end

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    # new start
    @torch.no_grad()
    def _vb_terms(self, x_start, x_t, t, eps_pred, model_var_values):
        frozen_out = torch.cat([eps_pred.detach(), model_var_values], dim=1)
        out = self.train_diffusion._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=x_start, x_t=x_t, t=t, clip_denoised=False
        )["output"]
        if str(self.train_diffusion.loss_type).endswith("RESCALED_MSE"):
            out = out * self.train_diffusion.num_timesteps
        return out
    
    def forward(self, target, z, mask=None,
                    return_token_loss: bool = False,
                    return_cache: bool = False,
                    tau_scale: torch.Tensor = None):
        
        """
        target, z: [N, C]
        mask: [N] (1=有效)
        tau_scale: [N] 或 [N,1]；若提供，將學到的 τ 乘進訓練噪聲（B 法）
        """
        N, C = target.shape
        device = target.device

        t = torch.randint(0, self.train_diffusion.num_timesteps, (N,), device=device)
        eps_true = torch.randn_like(target)                               # 這就是 ε_true
        if tau_scale is not None:
            while tau_scale.dim() < 2:
                tau_scale = tau_scale.unsqueeze(-1)
            eps_true = eps_true * tau_scale                               # ε' = τ * ε_true

        x_t = self.train_diffusion.q_sample(target, t, noise=eps_true)

        model_out = self.net(x_t, t, dict(c=z))
        eps_pred, model_var = torch.split(model_out, C, dim=1)

        mse = _mean_flat((eps_true - eps_pred) ** 2)                      # [N]
        vb  = self._vb_terms(target, x_t, t, eps_pred, model_var)         # [N]
        loss_token = mse + vb

        if mask is not None:
            loss = (loss_token * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss_token.mean()

        if not (return_token_loss or return_cache):
            return loss

        ret = {"loss": loss}
        if return_token_loss:
            ret["token_loss"] = loss_token
        if return_cache:
            ret.update({"t": t, "noise": eps_true, "x_t": x_t})
        return ret
    
    @torch.no_grad()
    def sample_pairwise(self, z_cond, z_uncond, tau, cfg):
        """
        兩兩成對（cond/uncond）取樣，支援 tau/cfg 為標量或逐樣本（或逐 token 已收斂到選中集合）的向量 
        - z_cond, z_uncond: [N, D]
        - tau: float 或 [N] / [N,1]
        - cfg: float 或 [N] / [N,1]
        回傳 cond 分支的樣本（因 forward_with_cfg 兩半會一致） 
        """
        assert z_cond.shape == z_uncond.shape
        N = z_cond.shape[0]
        z_pair = torch.cat([z_cond, z_uncond], dim=0)  # [2N, D]

        # 準備 noise 與 temperature（允許逐樣本）
        noise = torch.randn(N, self.in_channels, device=z_pair.device)
        noise = torch.cat([noise, noise], dim=0)       # [2N, C]
        if not torch.is_tensor(tau):
            tau = torch.full((N, 1), float(tau), device=z_pair.device)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        tau_pair = torch.cat([tau, tau], dim=0)        # [2N, 1]

        # 準備 cfg：允許標量或 [N]；擴成 [2N] 傳進 forward_with_cfg
        if not torch.is_tensor(cfg):
            cfg = torch.full((N,), float(cfg), device=z_pair.device)
        cfg = cfg.view(N)                               # [N]
        cfg_pair = torch.cat([cfg, cfg], dim=0)         # [2N]

        model_kwargs = dict(c=z_pair, cfg_scale=cfg_pair)  # 允許逐樣本 cfg
        sampled = self.gen_diffusion.p_sample_loop(
            self.net.forward_with_cfg,
            noise.shape, noise, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, temperature=tau_pair
        )
        return sampled[:N]

    # new end

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x) # token
        t = self.time_embed(t) # time step
        c = self.cond_embed(c) # z

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
