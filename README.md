# DL Project – Autoregressive Image Generation (Enhanced MAR)

## 1) Brief introduction
This repo extends the paper **“Autoregressive Image Generation without Vector Quantization”** by replacing the fixed token generation schedule with learnable policies and by making guidance hyper-parameters adaptive.  

The model keeps the **MAE-style encoder/decoder** and **diffusion loss backbone** intact by default, while adding lightweight heads that decide:
- Which tokens to generate
- How many per step
- What τ/CFG to use (globally or per token)

Everything is **ablation-friendly** and can be toggled with simple flags.

---

## 2) What’s new vs. the original paper

### Learned schedule (Order + Group)
- **Original**: cosine decay on the masked ratio; fixed/random ordering.  
- **Ours**:  
  - A token selection policy learns which tokens to reveal next.  
  - A group size head learns how many to reveal each step.  
  - Training uses a *differentiable top-k fraction*; inference uses *hard top-k*.  

### Adaptive τ (temperature)
- **Original**: fixed sampling temperature.  
- **Ours**:  
  - A τ head predicts temperature globally or per token.  
  - During training, τ scales the diffusion training noise (*reparameterization*), so the main diffusion loss drives τ learning with zero extra forward passes.  

### Adaptive CFG (classifier-free guidance scale)
- **Original**: fixed CFG schedule.  
- **Ours**:  
  - A CFG head predicts guidance globally or per token.  
  - Training uses an **ε-consistency loss** with the ground-truth diffusion noise (ϵ_true) to supervise the guided prediction.  

---

## 3) Detailed highlights

### Which tokens first (Order Policy)
- A token-wise scoring head ranks masked tokens.  
- Training uses a **soft top-k fraction (`soft_topk_frac`)** to keep selection differentiable and stable.  
- A **small size penalty** encourages the model to pick about *k* tokens (where *k* is predicted by the group head).  

### How many at a time (Group Size Head)
- A scalar per image predicts the next step’s *k* as a ratio of remaining masked tokens (bounded to **[ρ_min, ρ_max]**).  
- This makes the schedule adaptive to **content difficulty** rather than following cosine decay.  

### Adaptive τ (global or token-level)
- **Global head**: outputs one τ per image.  
- **Token head**: outputs a τ per token.  

Training uses noise reparameterization:  
- ϵ′ = τ ⋅ ϵ_true  
- Then sample x_t with ϵ′.  

The **standard diffusion loss** w.r.t. ϵ̂ backpropagates through τ — *no extra loss or passes needed*.  

### Adaptive CFG (global or token-level)
- Predicts **w (CFG scale)** per image or per token.  
- Uses an **ε-consistency objective**:  
  - With frozen diffusion MLP compute (ϵ_c, ϵ_u) (cond/uncond).  
  - Form: ϵ_g = ϵ_u + w (ϵ_c − ϵ_u).  
  - Minimize ‖ϵ_g − ϵ_true‖² only w.r.t. the CFG head.  

---

## Ablation-ready inference
- One parameter `--infer_mode` controls sampling behavior:  
  - **baseline**  
  - **policy only** ([1,2])  
  - **guidance only** ([3,4])  
  - **both**  

---

## Freezing made simple
- `--freeze_backbone`: freeze the entire MAR except heads, or train everything.  
- `--train_diffloss`: choose whether to train the diffusion MLP or keep it frozen.  

> **Note**: No changes to your **MAE blocks** or **diffusion MLP** are required to start — new capabilities live in small heads and step embeddings.
