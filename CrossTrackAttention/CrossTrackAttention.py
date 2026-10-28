import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class Cfg:
    H: int = 360                 # working resolution (DAVIS is 480x854, same aspect)
    W: int = 640
    # FastKLT
    fast_thresh: int = 20
    grid_rows: int = 12          # at most one FAST seed per grid cell per detection
    grid_cols: int = 20
    border: int = 12
    detect_every: int = 6        # fixed re-detection schedule, so the teacher can label every seed ahead of time
    min_remaining: int = 8       # no detection in the last few frames of a clip
    max_tracks: int = 256
    min_dist: float = 10.0       # a new seed must be this far from every live track
    lk_win: int = 21
    lk_levels: int = 3
    oob_margin: float = 2.0
    # supervision / metrics
    delta_valid: float = 6.0     # px: a track-frame is correct if teacher-visible and within this of the teacher
    r_max: float = 16.0          # px: larger KLT errors are only rejected, not corrected
    # module
    window: int = 16
    d_model: int = 128
    heads: int = 4
    depth: int = 3               # (causal time block, kNN space block) pairs
    knn: int = 16
    iters_image: int = 2         # CoTracker-style refinement iterations, image arm
    max_window_tracks: int = 384

cfg = Cfg()
STRIDE, CORR_LEVELS, CORR_R = 4, 4, 3     # CoTracker3's fnet stride, pyramid levels, correlation radius
PATCH = torch.stack(torch.meshgrid(torch.arange(-CORR_R, CORR_R + 1), torch.arange(-CORR_R, CORR_R + 1)), -1).to(torch.float32) # (49, 2)
device = "cuda" if torch.cuda.is_available() else "cpu"
WH = torch.tensor([cfg.W, cfg.H], device=device, dtype=torch.float32)

class MLP(nn.module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__(nn.Linear(d_in, d_hidden),
                         nn.GELU(),
                         nn.Linear(d_hidden, d_out))
        
class CausalTimeBlock(nn.Module):
    def __init_(self, d, heads, windows):
        super().__init__()
        self.h = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.rel_bias= nn.Parameter(torch.zeros(heads, windows))
        self.mlp = MLP(d, 4*d, d)
        
    def forward(self, x, exists):
        B, W, N, d = x.shape # B: batch size, W: window size, N: number of tracks, d: feature dimension
        h = x.permute(0, 2, 1, 3).reshape(B*N, W, d) # Reshape to (B*N, W, d) for attention computation
        e = exists.permute(0, 2, 1).reshape(B*N, W)
        # Compute Q, K, V matrices for attention, reshaping to (3, B*N, h, W, d/h) for multi-head attention
        q, k, v = self.qkv(self.norm1(h)).view(B * N, W, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        
        i = torch.arange(W, device=x.device) 
        lag = i[None, :] - i[:, None] # Compute lag matrix for causal attention
        allowed = ((lag <= 0) & e[:, None, :]) | torch.eye(W, device=x.device)[None, :, :].bool() # Allow self-attention and causal masking
        # Compute relative positional bias for attention, masking out disallowed positions with -inf which will be ignored in softmax during layer normalization
        mask = self.rel_bias[:, lag.clamp(min=0)][None].masked_fill(~allowed[:, None], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # Compute attention output
        h = h + self.proj(out.permute(1, 3, 0, 2).reshape(B*N, W, d)) # Add attention output to input (residual connection)
        h = h + self.mlp(self.norm2(h)) # Add MLP output to input (residual connection)
        return h.reshape(B, N, W, d).permute(0, 2, 1, 3) # Reshape back to original dimensions (B, W, N, d)
  
def posenc(x, min_deg, max_deg):
    """CoTracker3's Fourier encoding: [x, sin(2^k x), cos(2^k x)], k in [min_deg, max_deg)."""
    scales = 2.0 ** torch.arange(min_deg, max_deg, device=x.device, dtype=x.dtype)
    xb = (x[..., None, :] * scales[:, None]).flatten(-2)
    return torch.cat([x, torch.sin(torch.cat([xb, xb + 0.5 * math.pi], -1))], -1)

def sample_patches(fmap, xy):
    """fmap (B, C, h, w), xy (B, N, 2) in feature pixels -> (B, N, 49, C). Bilinear, border padding, align_corners (as CoTracker3)."""
    _, _, h, w = fmap.shape
    p = xy[:, :, None, :] + PATCH
    grid = torch.stack([2 * p[..., 0] / (w - 1) - 1, 2 * p[..., 1] / (h - 1) - 1], -1)
    out = F.grid_sample(fmap, grid.to(fmap.dtype), mode="bilinear", padding_mode="border", align_corners=True)
    return out.permute(0, 2, 3, 1)

class CorrMLP(nn.Module):
    def __init__(self, patch=49, hidden=384, out=256):
        super().__init__()
        self.fc1 = nn.Linear(patch * patch, hidden)   # names match CoTracker's Mlp
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def corr_features(corr_mlp, pyr, tmpl, frame, xy):
    """tmpl (B, N, L, 49, C) L2-normalised; xy (B, W, N, 2) px -> (B, W, N, L*256)."""
    B, W, N, _ = xy.shape
    embs = []
    with torch.autocast(xy.device.type, dtype=torch.float16, enabled=xy.is_cuda):
        for l in range(CORR_LEVELS):
            tgt = sample_patches(pyr[l][frame.flatten()],
                                 xy.reshape(B * W, N, 2) / (STRIDE * 2 ** l))
            tgt = tgt.view(B, W, N, 49, -1)
            corr = torch.einsum("bwnpc,bnqc->bwnpq", tgt, tmpl[:, :, l])
            embs.append(corr_mlp(corr.flatten(-2)))           # (B, W, N, 256)
    return torch.cat(embs, -1).float()
  
class KnnSpaceBlock(nn.Module):
    def __init__(self, d, heads, k):
        super().__init__()
        self.h, self.k = heads, k
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d) # Layer normalization for input and output features
        # Layer normalization ensures that attended features have consistent scale and distribution, which stabilizes training and improves convergence.
        self.q, self.kv, self.proj = nn.Linear(d, d), nn.Linear(d, 2*d), nn.Linear(d, d)
        self.rel = MLP(34, 64, d + heads) # relational MLP for computing relative positional embeddings based on fourier encodings
        self.mlp = MLP(d, 4*d, d)
        
    def forward(self, x, xy, exists):
        B, W, N, d = x.shape
        H, dh, k = self.h, d // self.h, min(self.k, N)
        xs, ps, es = x.reshape(B*W, N, d), xy.reshape(B*W, N, 2), exists.reshape(B*W, N)
        dist = torch.cdist(ps, ps).masked_fill(~(es[:, :, None] & es[:, None, :]), float("inf")) # Compute pairwise distances and mask out non-existent tracks
        nd, nidx = dist.topk(k, largest=False) # Get k nearest neighbors based on distance
        bidx = torch.arange(B*W, device=x.device)[:, None, None] # Batch indices for gathering neighbors
        hn = self.norm1(xs[bidx, nidx]) # Normalize neighbor features
        kv = self.kv(hn)[bidx, nidx]  # Compute key and value matrices for attention from neighbor features
        r = self.rel(torch.fft.fft2(ps[bidx, nidx], dim=(-2, -1)).real) # Compute relative positional embeddings using Fourier transform
        kn = kv[..., :d].reshape(B*W, N, k, H, dh)
        vn = (kv[..., d:] + r[..., None, :H]).reshape(B*W, N, k, H, dh) # Add relative positional embeddings to value matrix
        q = self.q(hn).view(B*W, N, H, dh) # Compute query matrix from normalized neighbor features
        logits = (q[:, :, None, :, :] * kn[:, :, :, :, :]).sum(-1) / math.sqrt(dh) # Compute attention logits
        logits = logits.masked_fill(~(es[:, :, None] & es[bidx, nidx]), float("-inf")) # Mask out logits for non-existent neighbors
        attn = F.softmax(logits, dim=2) # Compute attention weights
        out = (attn[..., None] * vn).sum(2) # Compute attention output by weighted sum of value matrix
        xs = xs + self.proj(out)
        xs = xs + self.mlp(self.norm2(xs)) # Add MLP output to input (residual connection)
        return xs.reshape(B, W, N, d)

def motion_features(pos, exists, WH):
    """pos (B, W, N, 2) px, exists (B, W, N) bool -> (B, W, N, 84)."""
    valid = (exists[:, 1:] & exists[:, :-1]).unsqueeze(-1) # both endpoints alive
    d = (pos[:, 1:] - pos[:, :-1]) * valid # zero across birth/death
    fwd = F.pad(d, (0, 0, 0, 0, 0, 1))     # displacement out of frame t
    bwd = F.pad(d, (0, 0, 0, 0, 1, 0))     # displacement into frame t
    return posenc(torch.cat([fwd, bwd], -1) / WH, 0, 10)

def motion_features(pos, exists, WH):
    """pos (B, W, N, 2) px, exists (B, W, N) bool -> (B, W, N, 42). Causal: uses only P_t - P_{t-1}."""
    valid = (exists[:, 1:] & exists[:, :-1]).unsqueeze(-1)      # both endpoints alive
    d = (pos[:, 1:] - pos[:, :-1]) * valid / WH                    # normalise per axis before encoding
    bwd = F.pad(d, (0, 0, 0, 0, 1, 0))                             # displacement into frame t (0 at t=0)
    return posenc(bwd, 0, 10)


MOTION_DIM = 42

class TrackRefiner(nn.Module):
    def __init__(self, use_image, cross_track=True, d=cfg.d_model, heads=cfg.heads, depth=cfg.depth):
        super().__init__()
        self.use_image, self.cross_track = use_image, cross_track
        self.iters = cfg.iters_image if use_image else 1
        self.inp, self.delta_in = nn.Linear(MOTION_DIM, d), nn.Linear(42, d)
        self.corr_mlp = CorrMLP() if use_image else None
        self.corr_in = nn.Linear(CORR_LEVELS * 256, d) if use_image else None
        self.time = nn.ModuleList([CausalTimeBlock(d, heads, cfg.window) for _ in range(depth)])
        self.space = nn.ModuleList([KnnSpaceBlock(d, heads, cfg.knn) for _ in range(depth)]) if cross_track else None
        self.norm = nn.LayerNorm(d)
        self.head_delta, self.head_logit = nn.Linear(d, 2), nn.Linear(d, 3)   # vis, conf(corrected), conf(raw)
        nn.init.trunc_normal_(self.head_delta.weight, std=1e-3)
        nn.init.zeros_(self.head_delta.bias)

    def forward(self, b, pyr=None, tmpl=None):
        ex, raw = b["exists"], b["raw"]
        delta, outs = torch.zeros_like(raw), [] # corrections, outputs
        for _ in range(self.iters):
            delta = delta.detach()
            pos = raw + delta
            x = self.inp(motion_features(pos, ex, WH)) + self.delta_in(posenc(delta / WH, 0, 10))
            if self.use_image:
                x = x + self.corr_in(corr_features(self.corr_mlp, pyr, tmpl, b["frame"], pos))
            for i in range(len(self.time)):
                x = self.time[i](x, ex)
                if self.cross_track:
                    x = self.space[i](x, raw, ex)
            h = self.norm(x)
            delta = delta + self.head_delta(h).float() * STRIDE
            outs.append((delta, self.head_logit(h).float()))
        return outs
    
    
        