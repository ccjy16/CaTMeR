import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PVTPatchEmbed(nn.Module):

    def __init__(self, img_size=224, kernel_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = (kernel_size - stride + 1) // 2

        H = (img_size[0] + 2 * self.padding - kernel_size) // stride + 1
        W = (img_size[1] + 2 * self.padding - kernel_size) // stride + 1
        self.H = H
        self.W = W
        self.num_patches = H * W

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SRAttention(nn.Module):

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn

class PVTBlock(nn.Module):

    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.
        )

    def forward(self, x, H, W):
        attn_output, attn_weights = self.attn(self.norm1(x), H, W)
        x = x + attn_output

        x = x + self.mlp(self.norm2(x), H, W)
        return x, attn_weights

class PVTEncoderStage(nn.Module):

    def __init__(self, dim, depth, num_heads, sr_ratio=1, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            PVTBlock(dim, num_heads, sr_ratio, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        attn_weights_list = []
        for blk in self.blocks:
            x, attn_w = blk(x, H, W)
            attn_weights_list.append(attn_w)

        x = self.norm(x)
        final_attn = attn_weights_list[-1]
        token_importance = final_attn.mean(dim=1).mean(dim=1)
        return x, token_importance

class PVTMultiGranularityVisualEncoder(nn.Module):

    HEAD_PREFIXES = ('global_proj.', 'local_proj.', 'stage_weights', 'final_proj.')

    def __init__(self, img_size, local_k, output_dim):
        super().__init__()
        embed_dims = (64, 128, 320, 512)
        num_heads = (1, 2, 5, 8)
        depths = (3, 4, 6, 3)
        sr_ratios = (8, 4, 2, 1)
        mlp_ratios = (8, 8, 4, 4)
        self.img_size = img_size
        self.output_dim = output_dim
        self.local_k = local_k
        self.sr_ratios = tuple(sr_ratios)
        self.num_stages = 4

        self.patch_embeds = nn.ModuleList()
        grid_size = img_size
        for stage in range(self.num_stages):
            patch = PVTPatchEmbed(
                img_size=grid_size, kernel_size=7 if stage == 0 else 3,
                stride=4 if stage == 0 else 2,
                in_chans=3 if stage == 0 else embed_dims[stage - 1],
                embed_dim=embed_dims[stage],
            )
            self.patch_embeds.append(patch)
            grid_size = (patch.H, patch.W)

        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.patch_embeds[i].num_patches, embed_dims[i]))
            for i in range(self.num_stages)
        ])
        self.pos_drops = nn.ModuleList([nn.Dropout(p=0.0) for _ in range(self.num_stages)])


        self.stages = nn.ModuleList([
            PVTEncoderStage(
                dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i]
            )
            for i in range(self.num_stages)
        ])

        self.global_proj = nn.ModuleList([
            nn.Linear(embed_dims[i], output_dim // 2) for i in range(self.num_stages)
        ])
        self.local_proj = nn.ModuleList([
            nn.Linear(embed_dims[i] * local_k, output_dim // 2) for i in range(self.num_stages)
        ])
        self.stage_weights = nn.Parameter(torch.ones(self.num_stages) / self.num_stages)
        self.final_proj = nn.Linear(output_dim, output_dim)

        self.apply(self._init_weights)
        self._init_pos_embeds()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_pos_embeds(self):
        for i, pos_embed in enumerate(self.pos_embeds):
            nn.init.trunc_normal_(pos_embed, std=.02)

    def extract_local_features(self, x, token_importance, k, H=None, W=None, sr_ratio=1):
        B, N, C = x.shape
        if H is None or W is None or H * W != N:
            raise ValueError("PVT local feature grid must match the original token grid")

        if sr_ratio > 1:
            x = F.avg_pool2d(
                x.transpose(1, 2).reshape(B, C, H, W),
                kernel_size=sr_ratio, stride=sr_ratio,
            ).flatten(2).transpose(1, 2)
        if token_importance.shape != x.shape[:2]:
            raise ValueError("PVT attention grid does not match the local region grid")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 < k <= token_importance.shape[1]:
            raise ValueError("PVT local_k exceeds the effective attention grid capacity")
        topk_indices = torch.topk(token_importance, k, dim=1).indices
        local_features = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))
        return local_features.reshape(B, -1)

    def forward(self, x):
        batch_size = x.shape[0]
        features_list = []

        for i in range(self.num_stages):
            if i == 0:
                x, H, W = self.patch_embeds[i](x)
            else:
                x = x.permute(0, 2, 1).reshape(batch_size, -1, H, W)
                x, H, W = self.patch_embeds[i](x)

            x = x + self.pos_embeds[i][:, :x.size(1), :]
            x = self.pos_drops[i](x)

            if self.training and torch.is_grad_enabled() and x.requires_grad:
                x, token_importance = checkpoint(self.stages[i], x, H, W, use_reentrant=True)
            else:
                x, token_importance = self.stages[i](x, H, W)

            global_feat = x.mean(dim=1)
            global_feat = self.global_proj[i](global_feat)

            local_feat = self.extract_local_features(
                x, token_importance, self.local_k, H=H, W=W, sr_ratio=self.sr_ratios[i]
            )
            local_feat = self.local_proj[i](local_feat)

            stage_feat = torch.cat([global_feat, local_feat], dim=1)
            features_list.append(stage_feat)

        stacked_features = torch.stack(features_list, dim=1)
        weights = F.softmax(self.stage_weights, dim=0).view(1, -1, 1)
        aggregated = (stacked_features * weights).sum(dim=1)
        features = self.final_proj(aggregated)
        return features


def remap_pvt_state_dict(state_dict):
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("module.", "").replace("backbone.", "")
        name = re.sub(
            r"^patch_embed(\d+)(.*)",
            lambda match: f"patch_embeds.{int(match.group(1)) - 1}{match.group(2)}",
            name,
        )
        name = re.sub(
            r"^block(\d+)\.(\d+)(.*)",
            lambda match: (
                f"stages.{int(match.group(1)) - 1}.blocks."
                f"{match.group(2)}{match.group(3)}"
            ),
            name,
        )
        name = re.sub(
            r"^norm(\d+)\.(\w+)$",
            lambda match: f"stages.{int(match.group(1)) - 1}.norm.{match.group(2)}",
            name,
        )
        name = re.sub(
            r"^pos_embed(\d+)(.*)",
            lambda match: f"pos_embeds.{int(match.group(1)) - 1}{match.group(2)}",
            name,
        )
        mapped[name] = value
    return mapped

def validate_pvt_load_result(missing_keys, unexpected_keys):
    allowed_missing = (
        *PVTMultiGranularityVisualEncoder.HEAD_PREFIXES,
        "pos_embeds.",
    )
    missing_backbone = [
        name for name in missing_keys
        if not name.startswith(allowed_missing)
    ]
    if missing_backbone:
        preview = ", ".join(missing_backbone[:5])
        raise RuntimeError(f"PVT checkpoint is missing backbone parameters: {preview}")
    return {
        "missing_backbone": missing_backbone,
        "missing_heads": list(missing_keys),
        "unexpected": list(unexpected_keys),
    }

def load_pvt_backbone(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PVT checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(remap_pvt_state_dict(source), strict=False)
    return validate_pvt_load_result(
        incompatible.missing_keys, incompatible.unexpected_keys
    )
