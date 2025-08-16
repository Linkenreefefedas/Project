# all new
import torch
import torch.nn as nn
import math

# ---- Utilities ----

def sinusoidal_embedding(n_positions: int, dim: int, device=None):
    position = torch.arange(n_positions, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class StepEmbedding(nn.Module):
    """Embed the current generation step k and total steps K into a vector."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, step: int, total_steps: int, batch: int, device):
        k = torch.tensor([step], device=device).long()
        K = torch.tensor([total_steps], device=device).long()
        pe_k = sinusoidal_embedding(int(k.max().item() + 2), self.emb_dim, device=device)[k]
        pe_K = sinusoidal_embedding(int(K.max().item() + 2), self.emb_dim, device=device)[K]
        pe = torch.cat([pe_k, pe_K], dim=-1)  # [1, 2D]
        pe = self.proj(pe).repeat(batch, 1)    # [B, D]
        return pe

# ---------- ordering & grouping ----------
class TokenPolicy(nn.Module):
    """
    Score each masked token with a scalar priority.
    Higher score = generate earlier.
    """
    def __init__(self, token_dim: int, cond_dim: int, hidden: int = 512):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, z_tokens: torch.Tensor, mask: torch.Tensor, step_embed: torch.Tensor):
        """
        z_tokens: [B, S, D] decoder features
        mask:     [B, S] (1 = masked, 0 = already generated/visible)
        step_embed: [B, D]
        return: scores [B, S]
        """
        B, S, D = z_tokens.shape
        vis_mask = (mask == 0).float().unsqueeze(-1)  # [B,S,1]
        msk_mask = (mask == 1).float().unsqueeze(-1)  # [B,S,1]
        eps = 1e-6
        vis_ctx = (z_tokens * vis_mask).sum(dim=1) / (vis_mask.sum(dim=1) + eps)  # [B,D]
        msk_ctx = (z_tokens * msk_mask).sum(dim=1) / (msk_mask.sum(dim=1) + eps)  # [B,D]
        cond = vis_ctx + msk_ctx + step_embed                                       # [B,D]

        cond_h = self.cond_mlp(cond)            # [B,H]
        tok_h  = self.token_mlp(z_tokens)       # [B,S,H]
        h = tok_h + cond_h.unsqueeze(1)         # broadcast -> [B,S,H]
        scores = self.out(h).squeeze(-1)        # [B,S]
        return scores


class GroupSizeHead(nn.Module):
    """
    Predict the fraction r in (min_ratio, max_ratio) of remaining masked tokens to generate in this step.
    """
    def __init__(self, cond_dim: int, min_ratio: float = 0.01, max_ratio: float = 0.5):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        hidden = max(256, cond_dim // 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_tokens: torch.Tensor, mask: torch.Tensor, step_embed: torch.Tensor):
        """
        z_tokens: [B,S,D], mask: [B,S]
        returns r in (min_ratio, max_ratio), shape [B]
        """
        msk = (mask == 1).float().unsqueeze(-1)
        eps = 1e-6
        ctx = (z_tokens * msk).sum(dim=1) / (msk.sum(dim=1) + eps)  # [B,D]
        ctx = ctx + step_embed                                      # [B,D]
        r = torch.sigmoid(self.mlp(ctx)).squeeze(-1)                # [B]
        r = self.min_ratio + (self.max_ratio - self.min_ratio) * r
        return r

def soft_topk_frac(scores: torch.Tensor,
                   k_float: torch.Tensor,
                   valid_mask: torch.Tensor,
                   tau: float = 0.1):
    """
    連續 k 的可微選位器（讓「一次找幾個」端到端可學） 
    - scores: [B,S]，order 分數
    - k_float: [B]，連續 k
    - valid_mask: [B,S]，1=仍可選（masked）
    - tau: 溫度，小→近似 hard

    作法：在有效位置排序，取 floor/ceil(k) 的分界做線性內插門檻，再用 sigmoid 產生 soft mask 
    回傳：
      soft_mask: [B,S] ∈ [0,1]
      size_penalty: 樣本平均的 (∑soft - k)^2，幫助學到合適批量大小
    """
    B, S = scores.shape
    very_neg = torch.finfo(scores.dtype).min / 4
    scr = scores.masked_fill(valid_mask == 0, very_neg)

    vals, _ = torch.sort(scr, dim=1, descending=True)
    vlen = valid_mask.sum(dim=1).clamp(min=1)

    k0 = torch.floor(k_float).clamp(min=1, max=vlen)
    k1 = torch.clamp(k0 + 1, max=vlen)
    alpha = (k_float - k0).clamp(0, 1)

    idx0 = (k0.long() - 1).clamp(min=0)
    idx1 = (k1.long() - 1).clamp(min=0)

    t0 = vals.gather(1, idx0.unsqueeze(1)).squeeze(1)   # [B]
    t1 = vals.gather(1, idx1.unsqueeze(1)).squeeze(1)   # [B]
    thr = (1 - alpha) * t0 + alpha * t1                 # [B]

    soft = torch.sigmoid((scr - thr.unsqueeze(1)) / tau) * valid_mask
    size_pen = ((soft.sum(dim=1) - k_float) ** 2).mean()
    return soft, size_pen


# ---------- adaptive tau/cfg ----------
class TokenScalarHead(nn.Module):
    """
    對每個 token 預測一個標量（經 Sigmoid 後映射到 [low, high]） 
    可用於 τ 或 CFG（以不同範圍實例化） 
    """
    def __init__(self, token_dim: int, cond_dim: int, low: float, high: float, hidden: int = 512):
        super().__init__()
        self.low, self.high = low, high
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, z_tokens: torch.Tensor, step_embed: torch.Tensor):
        # z_tokens: [B,S,D], step_embed: [B,D]
        h_tok = self.token_mlp(z_tokens)                   # [B,S,H]
        h_cond = self.cond_mlp(step_embed).unsqueeze(1)    # [B,1,H]
        y = self.out(h_tok + h_cond).squeeze(-1)           # [B,S]
        y = torch.sigmoid(y) * (self.high - self.low) + self.low
        return y  # [B,S] in [low, high]
    
class GlobalScalarHead(nn.Module):
    """
    對整個步驟預測一個全域標量（經 Sigmoid 後映射到 [low, high]） 
    """
    def __init__(self, cond_dim: int, low: float, high: float):
        super().__init__()
        self.low, self.high = low, high
        hidden = max(256, cond_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_tokens: torch.Tensor, step_embed: torch.Tensor):
        # 用 token 平均作為全域上下文
        ctx = z_tokens.mean(dim=1) + step_embed            # [B,D]
        y = self.net(ctx).squeeze(-1)                      # [B]
        y = torch.sigmoid(y) * (self.high - self.low) + self.low
        return y  # [B] in [low, high]