from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss
from models.policy import TokenPolicy, GroupSizeHead, StepEmbedding, soft_topk_frac,TokenScalarHead, GlobalScalarHead


#MAR (Masked Autoregressive Model) 主模型架構，包含不同規模的模型建構函數（如 mar_base, mar_large, mar_huge） 
#ok AR生成z
'''
輸入：已經過 VAE 編碼成 latent space 的影像（形狀 [B, latent_C, latent_H, latent_W]） 

編碼器（MAE encoder）：接收部份可見 tokens，經過 Vision Transformer 編碼剩餘可見資訊 

解碼器（MAE decoder）：補上 Mask Token，再經 Transformer 還原序列資訊，並傳給 Diffusion 模塊去預測被 Mask 的 token 

Diffusion Loss：針對被 Mask 的位置，用擴散模型預測對應的 latent 表示，計算重建損失 

生成模式（Sampling）：逐輪反覆採樣未生成的 token，類似 MaskGIT/MAGE 的 iterative decoding 
'''
def mask_by_order(mask_len, order, bsz, seq_len):
    # 依據隨機生成的順序 order，指定前 mask_len 個位置為 Mask
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

# new start
def _mean_flat(x: torch.Tensor):
    return x.view(x.size(0), -1).mean(dim=1)
# new end

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        #將 VAE latent 分割成序列
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        #將 token 序列還原成 [B, C, H, W] latent 格式
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        # 為每張圖生成隨機的生成順序，確保 iterative decoding 時 token 的生成順序隨機化
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        '''
        將 patch 序列投影到 encoder_embed_dim
        
        前面加上 buffer tokens（這裡用於存放 class embedding 或其他條件資訊）
        
        加入位置編碼
        
        移除 Mask 的 token（只保留可見部分）
        
        經過多層 Transformer Block 得到壓縮特徵
        '''
        '''
        接收被 patchify 後的 latent token（部分被 mask），
        將可見 token + 類別條件（class embedding）送入編碼器，
        提取可用的上下文特徵，為後續重建做準備 
        '''
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        '''
        投影到 decoder_embed_dim
        
        Mask 的位置用 mask_token 補回
        
        加入位置編碼
        
        經 Transformer 還原完整序列
        
        加入 diffusion position embedding
        （這是要餵進 DiffLoss 預測 latent 的位置資訊）
        '''
        '''
        接收編碼器輸出的可見 token 特徵，
        重新插入 mask token 補足完整序列，
        經過解碼器恢復所有位置的特徵（包括被 mask 的位置），
        最後加上 diffusion 的位置編碼，準備給 DiffLoss 做預測 
        '''
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        '''
        只針對mask部分算loss
        對每個 Mask 的位置，使用 DiffLoss 計算預測與 Ground Truth latent 之間的損失

        支援 diffusion_batch_mul，等於同一個 token 會多次擴散取樣計算 loss，每個token算4次
        '''
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):
        '''
        訓練流程：
        
        class embedding
        
        patchify + masking
        
        MAE encoder → MAE decoder
        
        DiffLoss 計算 masked token 的 reconstruction loss
        '''
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        '''
        生成流程：
        
        全部位置初始化為 Mask
        
        每輪生成部份 token（根據 cosine decay 的 mask ratio）
        
        Class embedding + CFG（classifier-free guidance）
        
        MAE encoder + decoder
        
        用 DiffLoss 的 sample 產生 token latent
        
        更新 token，直到全部生成完成
        
        unpatchify 回 latent 格式
        '''
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

    # new start
    def enable_learned_schedule(self, min_ratio=0.05, max_ratio=0.35, policy_tau=0.1, policy_loss_w=0.2, size_pen_w=1e-2):
        """
        同時啟用：
          1) 哪些 token 先生成（order）
          2) 一次生成幾個（group）
        """
        d_dec = self.decoder_blocks[0].norm1.normalized_shape[0]
        self.use_learned_schedule = True
        self.rho_min, self.rho_max = min_ratio, max_ratio
        self._policy_tau = policy_tau
        self._policy_w = policy_loss_w
        self._size_w = size_pen_w

        self.step_embed  = StepEmbedding(d_dec)
        self.policy_head = TokenPolicy(token_dim=d_dec, cond_dim=d_dec, hidden=max(256, d_dec // 2))
        self.group_head  = GroupSizeHead(cond_dim=d_dec, min_ratio=min_ratio, max_ratio=max_ratio)

    def disable_learned_schedule(self):
        self.use_learned_schedule = False

    def enable_learn_tau(self, mode="token"):
        """
        學 τ（B 法：把 τ 乘進訓練噪聲，靠主損驅動），mode in {'global','token'}
        """
        assert mode in ("global", "token")
        d_dec = self.decoder_blocks[0].norm1.normalized_shape[0]
        self.learn_tau = True
        self.tau_mode = mode
        self.step_embed_guid = StepEmbedding(d_dec)
        self.tau_global = GlobalScalarHead(cond_dim=d_dec, low=0.9, high=1.05)
        self.tau_token  = TokenScalarHead(token_dim=d_dec, cond_dim=d_dec, low=0.9, high=1.05)

    def disable_learn_tau(self):
        self.learn_tau = False

    def enable_learn_cfg(self, mode="token"):
        """
        學 CFG（用 ε_true 一致性），mode in {'global','token'}
        """
        assert mode in ("global", "token")
        d_dec = self.decoder_blocks[0].norm1.normalized_shape[0]
        self.learn_cfg = True
        self.cfg_mode = mode
        self.step_embed_guid = StepEmbedding(d_dec)
        self.cfg_global = GlobalScalarHead(cond_dim=d_dec, low=0, high=4)
        self.cfg_token = TokenScalarHead(token_dim=d_dec, cond_dim=d_dec, low=0, high=4)
     
    def disable_learn_cfg(self):
        self.learn_cfg = False

    def _get_step_embed(self, step:int, total_steps:int, bsz:int, device):
        return self.step_embed_policy(step=step, total_steps=total_steps, batch=bsz, device=device)

    # === 單次主損 + 可學策略 + 學 τ/CFG（全在這一個 forward 內完成） ===
    def forward_joint(self, imgs, labels, total_steps_hint: int = 64):
        """
        - 主損：一次 DiffLoss 前向（可選 τ 重參數化）
        - 策略：use_learned_schedule=True 同時啟用 order + group（連續 k）
        - 學 τ：B 法（把 τ 乘進噪聲），global/token
        - 學 CFG：用 ε_true 一致性，global/token（只更新 cfg head）
        - 注意：不做 class-drop，該邏輯已在 forward_mae_encoder 內處理
        """
        B = imgs.size(0); device = imgs.device
    
        # patchify + masking（取得 ground-truth latents 與 mask）
        x0_tok = self.patchify(imgs)                         # [B,S,Ctok]
        gt_latents = x0_tok.detach()
        orders = self.sample_orders(bsz=B)
        mask = self.random_masking(x0_tok, orders).to(torch.long)  # [B,S] 1=masked
    
        # cond features（MAE 凍結僅作前向）
        class_emb = self.class_emb(labels) if labels is not None else self.fake_latent.repeat(B, 1)
        enc = self.forward_mae_encoder(x0_tok, mask, class_emb)
        z_c = self.forward_mae_decoder(enc, mask)            # [B,S,D]
    
        # ===== 準備 τ 重參數化（B 法）：組 [N] tau_scale 傳給 DiffLoss.forward =====
        S = z_c.size(1); R = int(getattr(self, "diffusion_batch_mul", 1))
        if getattr(self, "learn_tau", False):
            step_emb_g = self.step_embed_guid(step=0, total_steps=total_steps_hint, batch=B, device=device)
            if self.tau_mode == "global":
                tau_map = self.tau_global(z_c, step_emb_g).unsqueeze(1).expand(-1, S)     # [B,S]
            else:
                tau_map = self.tau_token(z_c, step_emb_g)                                  # [B,S]
            tau_map = torch.where(mask.bool(), tau_map, torch.ones_like(tau_map))          # 非 masked 用 1
            tau_flat = tau_map.reshape(B*S).repeat(R)                                      # [N]
        else:
            tau_flat = None
    
        # ===== 一次 DiffLoss 前向：主損 + token 難度 + 回傳 ε_true / t / x_t =====
        flat_target = gt_latents.reshape(B*S, -1).repeat(R, 1)                             # [N,Ctok]
        flat_zc     = z_c.reshape(B*S, -1).repeat(R, 1)                                    # [N,D]
        flat_mask   = mask.reshape(B*S).repeat(R)                                          # [N]
        out = self.diffloss(target=flat_target, z=flat_zc, mask=flat_mask,
                            return_token_loss=True, return_cache=True,
                            tau_scale=tau_flat)
        loss_main = out["loss"]
        tok_loss = out["token_loss"].view(R, B*S).mean(0).view(B, S)                       # [B,S]
        valid = mask.to(tok_loss.dtype)
        difficulty = tok_loss * valid                                                      # 只看 masked
    
        total_loss = loss_main
    
        # ===== (1)+(2) 同時啟用：order + group（連續 k）=====
        if getattr(self, "use_learned_schedule", False):
            step_emb = self._get_step_embed(step=0, total_steps=total_steps_hint, bsz=B, device=device)
            scores = self.policy_head(z_c, mask, step_emb)                                  # [B,S]
            remaining = mask.sum(1).float()                                                 # [B]
            r = self.group_head(z_c, mask, step_emb)                                        # [B] in [rho_min,rho_max]
            k_float = (r * remaining).clamp(min=1.0)                                        # 連續 k
            soft, size_pen = soft_topk_frac(scores, k_float, valid_mask=(mask > 0), tau=self._policy_tau)
            # 期望難度（越難→越該被挑中； 也可改成 easy-first）
            order_loss = (soft * difficulty).sum() / soft.sum().clamp(min=1.0)
            total_loss = total_loss + self._policy_w * order_loss + self._size_w * size_pen
    
        # ===== (4) 學 CFG：用 ε_true 一致性（只更新 cfg head；不回傳到 MLP/MAE）=====
        if getattr(self, "learn_cfg", False):
            # 取得與主損同一批的 ε_true / t / x_t（保證一致）
            t_all = out["t"]                  # [N]
            eps_true_all = out["noise"]       # [N,Ctok]
            x_t_all = out["x_t"]              # [N,Ctok]
    
            # 準備 uncond 特徵（MAE 再走一次；enc/dec 凍結不更新）
            u_class = self.fake_latent.repeat(B, 1)
            enc_u = self.forward_mae_encoder(x0_tok, mask, u_class)
            z_u = self.forward_mae_decoder(enc_u, mask)                                     # [B,S,D]
    
            # 建 cfg_pred（global/token）
            step_emb_g = self.step_embed_guid(step=0, total_steps=total_steps_hint, batch=B, device=device)
            if self.cfg_mode == "global":
                cfg_map = self.cfg_global(z_c, step_emb_g).unsqueeze(1).expand(-1, S)       # [B,S]
            else:
                cfg_map = self.cfg_token(z_c, step_emb_g)                                   # [B,S]
    
            # flatten 並只在 masked 位置計算
            zc_flat = z_c.reshape(B*S, -1).repeat(R, 1)                                     # [N,D]
            zu_flat = z_u.reshape(B*S, -1).repeat(R, 1)                                     # [N,D]
            cfg_flat = cfg_map.reshape(B*S).repeat(R)                                       # [N]
            sel = flat_mask.nonzero(as_tuple=True)[0]                                       # [M]
            if sel.numel() > 0:
                # 用同一個 x_t/t 前向「凍結」的 MLP，取得 eps_c/eps_u，再與 ε_true 做一致性
                with torch.no_grad():
                    eps_c = self.diffloss.net(x_t_all[sel], t_all[sel], dict(c=zc_flat[sel]))[:, :gt_latents.size(2)]
                    eps_u = self.diffloss.net(x_t_all[sel], t_all[sel], dict(c=zu_flat[sel]))[:, :gt_latents.size(2)]
                delta = (eps_c - eps_u)                                                     # [M,Ctok] 常數
                cfg_sel = cfg_flat[sel].unsqueeze(1)                                        # [M,1]
                eps_g = eps_u + cfg_sel * delta
                cfg_loss = _mean_flat(eps_g - eps_true_all[sel]).mean()
                # 只讓 cfg head 收梯度
                total_loss = total_loss + cfg_loss
    
        return total_loss
    
    @torch.no_grad()
    def sample_tokens_joint(
        self,
        bsz,
        num_iter: int = 64,
        labels=None,
        default_tau: float = 1.0,
        default_cfg: float = 1.0,
        cfg_schedule: str = "linear",
        progress: bool = False,
        mode: str = "both",  # 'baseline' | 'policy' | 'guidance' | 'both'
    ):
        """
        推論（cond 決策） mode 控制消融：
          - 'baseline' : 不用學策略、不用學 τ/CFG 回到 cosine + 固定 τ/CFG 
          - 'policy'   : 只開 [1,2]（學順序 + 學批量），不用學 τ/CFG 
          - 'guidance' : 只開 [3,4]（學 τ + 學 CFG） 
          - 'both'     : 1~4 全開 
        """
        device = self.fake_latent.device
        S = self.seq_len
    
        use_policy   = (mode in ("policy", "both"))   and getattr(self, "use_learned_schedule", False)
        use_tau      = (mode in ("guidance", "both")) and getattr(self, "learn_tau", False)
        use_cfg      = (mode in ("guidance", "both")) and getattr(self, "learn_cfg", False)
    
        # 初始化
        mask   = torch.ones(bsz, S, device=device)                        # 1=尚未生成
        tokens = torch.zeros(bsz, S, self.token_embed_dim, device=device)
        orders = self.sample_orders(bsz)                                  # baseline 需要
    
        steps = range(num_iter)
        if progress:
            from tqdm import tqdm as _tqdm
            steps = _tqdm(steps)
    
        def cosine_mask_step(cur_mask: torch.Tensor, step: int):
            """原論文：cosine 遞減 mask，回傳 (sel_mask, mask_next)"""
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
            next_keep  = torch.floor(torch.tensor([S * mask_ratio], device=device))
            next_keep  = torch.maximum(torch.tensor([1.0], device=device),
                                       torch.minimum(cur_mask.sum(dim=-1, keepdims=True) - 1, next_keep))
            mask_next  = mask_by_order(next_keep[0], orders, bsz, S)      # [B,S]
            if step >= num_iter - 1:
                sel_mask = cur_mask.bool()
            else:
                sel_mask = torch.logical_xor(cur_mask.bool(), mask_next.bool())
            return sel_mask, mask_next
    
        for step in steps:
            cur_tokens = tokens.clone()
    
            # cond embedding（不 drop；policy 與 guidance 均以 cond 決策/預測）
            class_emb = self.class_emb(labels) if labels is not None else self.fake_latent.repeat(bsz, 1)
    
            # 是否需要 cfg 成對
            need_cfg_pair = (use_cfg or (default_cfg != 1.0))
    
            if need_cfg_pair:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                mask_in   = torch.cat([mask, mask], dim=0)
                class_in  = torch.cat([class_emb, self.fake_latent.repeat(bsz, 1)], dim=0)
            else:
                tokens_in, mask_in, class_in = tokens, mask, class_emb
    
            # MAE encoder/decoder（凍結，只前向）
            x = self.forward_mae_encoder(tokens_in, mask_in, class_in)
            z_full = self.forward_mae_decoder(x, mask_in)                 # [2B,S,D] 或 [B,S,D]
    
            # 取 cond/uncond 分支
            if need_cfg_pair:
                z_c, z_u = z_full[:bsz], z_full[bsz:]
            else:
                z_c = z_full
    
            # === 選哪些 & 選多少 ===
            valid  = (mask == 1)                                          # [B,S]
            remain = valid.sum(1)                                         # [B]
    
            if use_policy:
                # 同時開 [1,2]：learned order + learned group（以 cond 特徵決策）
                step_emb = self.step_embed_policy(step=step, total_steps=num_iter, batch=bsz, device=device)
                scores   = self.policy_head(z_c, valid, step_emb)         # [B,S]
                r        = self.group_head(z_c, valid, step_emb)          # [B] ∈ [rho_min,rho_max]
                k        = torch.round(r * remain.float()).clamp(min=1, max=remain).long()
                # 硬 top-k（只在 valid 中選）
                scr = scores.masked_fill(valid == 0, -1e9)
                idx = torch.argsort(scr, dim=1, descending=True)
                sel_mask = torch.zeros_like(valid, dtype=torch.bool)
                for b in range(bsz):
                    kk = int(max(1, min(int(k[b].item()), int(remain[b].item()))))
                    sel_mask[b, idx[b, :kk]] = True
                # 更新當前 mask
                mask[sel_mask] = 0
            else:
                # baseline：cosine + orders
                sel_mask, mask_next = cosine_mask_step(mask, step)
                mask = mask_next.clone()
    
            if sel_mask.sum() == 0:
                continue
            rows, cols = sel_mask.nonzero(as_tuple=True)
    
            # === 取本步 τ / CFG（token 或 global），或 baseline ===
            # τ：若 use_tau，從 τ 頭取；否則用 default_tau
            if use_tau:
                step_emb_g = self.step_embed_guid(step=step, total_steps=num_iter, batch=bsz, device=device)
                if getattr(self, "tau_mode", "token") == "global":
                    tau_map = self.tau_global(z_c, step_emb_g).unsqueeze(1).expand(-1, S)   # [B,S]
                else:
                    tau_map = self.tau_token(z_c, step_emb_g)                               # [B,S]
                tau_sel = tau_map[rows, cols]                                               # [N_sel]
            else:
                tau_sel = torch.full((rows.numel(),), float(default_tau), device=device)
    
            # CFG：若 use_cfg，從 cfg 頭取；否則 baseline（Muse 線性/常數）
            if use_cfg:
                if not need_cfg_pair:
                    # 安全性：cfg 需要 cond/uncond 成對，理論上這裡應該永遠不會進來
                    need_cfg_pair = True
                if getattr(self, "cfg_mode", "token") == "global":
                    cfg_map = self.cfg_global(z_c, step_emb_g).unsqueeze(1).expand(-1, S)   # [B,S]
                else:
                    cfg_map = self.cfg_token(z_c, step_emb_g)                               # [B,S]
                cfg_sel = cfg_map[rows, cols]                                               # [N_sel]
            else:
                # baseline：Muse 調度
                if cfg_schedule == "linear":
                    remain_min = remain.min()
                    cfg_iter = 1 + (default_cfg - 1) * (self.seq_len - remain_min) / self.seq_len
                elif cfg_schedule == "constant":
                    cfg_iter = default_cfg
                else:
                    cfg_iter = default_cfg
                cfg_sel = torch.full((rows.numel(),), float(cfg_iter), device=device)
    
            # === 取樣：有 CFG → pairwise；否則單分支（cond） ===
            if need_cfg_pair:
                zc_sel = z_c[rows, cols]
                zu_sel = z_u[rows, cols]
                sampled = self.diffloss.sample_pairwise(zc_sel, zu_sel, tau=tau_sel, cfg=cfg_sel)  # [N_sel,Ctok]
            else:
                zc_sel = z_c[rows, cols]
                # 若 diffloss.sample 支援 per-sample temperature（我們改過），可直接塞 tau_sel
                # 否則以平均值近似：
                if tau_sel.ndim == 1:
                    tau_used = tau_sel.mean().item() if tau_sel.numel() > 0 else float(default_tau)
                else:
                    tau_used = float(default_tau)
                sampled = self.diffloss.sample(zc_sel, temperature=tau_used, cfg=1.0)
    
            # 寫回 & 檢查完成
            cur_tokens[rows, cols] = sampled
            tokens = cur_tokens.clone()
            if mask.sum() == 0:
                break
    
        return self.unpatchify(tokens)


    # new end


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
