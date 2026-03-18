# models/backbones/transformers.py
# SOVEREIGN YOLOv5 Transformer Backbone Implementation
# Native support for Swin Transformer and EfficientFormer backbones

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Union
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], 
               self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, 
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition to feature map."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with window attention and shifted window attention."""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., 
                 attn_drop: float = 0., drop_path: float = 0., 
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.H = None
        self.W = None
    
    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling."""
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        # Pad if needed
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., 
                 attn_drop: float = 0., drop_path: Union[float, List[float]] = 0., 
                 norm_layer: nn.Module = nn.LayerNorm, downsample: Optional[nn.Module] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        # Calculate attention mask for SW-MSA
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer backbone for YOLOv5.
    
    Args:
        pretrain_img_size (int): Input image size for pretrained models. Default: 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.2
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    
    def __init__(self, pretrain_img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24), window_size: int = 7,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.2,
                 norm_layer: nn.Module = nn.LayerNorm, ape: bool = False,
                 patch_norm: bool = True, out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 frozen_stages: int = -1, use_checkpoint: bool = False, **kwargs):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (pretrain_img_size // patch_size) ** 2
        self.patch_resolution = (pretrain_img_size // patch_size, pretrain_img_size // patch_size)
        
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
        
        # Add a norm layer for each output
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.norm_layers = nn.ModuleList([norm_layer(num_features[i]) for i in self.out_indices])
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def init_weights(self, pretrained: Optional[str] = None):
        """Initialize the weights in backbone.
        
        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        if pretrained is None:
            self.apply(_init_weights)
        else:
            # Load pretrained weights
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Filter out unnecessary keys
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in self.state_dict() and v.shape == self.state_dict()[k].shape:
                    filtered_dict[k] = v
            
            self.load_state_dict(filtered_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward function."""
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        
        if self.ape:
            # Interpolate absolute position embedding
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed.reshape(1, self.patch_resolution[0], self.patch_resolution[1], -1).permute(0, 3, 1, 2),
                size=(H, W), mode='bicubic'
            ).permute(0, 2, 3, 1).flatten(1, 2)
            x = x.flatten(2).transpose(1, 2) + absolute_pos_embed
        else:
            x = x.flatten(2).transpose(1, 2)
        
        x = self.pos_drop(x)
        
        outs = []
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            
            if i in self.out_indices:
                norm_layer = self.norm_layers[self.out_indices.index(i)]
                out = norm_layer(x)
                out = out.view(-1, H, W, int(self.embed_dim * 2 ** i)).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keep layers frozen."""
        super().train(mode)
        self._freeze_stages()


class EfficientFormer(nn.Module):
    """EfficientFormer backbone for YOLOv5.
    
    Args:
        depths (list[int]): Number of blocks at each stage.
        embed_dims (list[int]): Embedding dimension at each stage.
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim at each stage.
        downsamples (list[bool]): Whether to downsample at each stage.
        num_classes (int): Number of classes. Default: 1000
        in_chans (int): Number of input image channels. Default: 3
        vit_num (int): Number of ViT blocks at the end. Default: 0
        resolution (int): Input resolution. Default: 224
        distillation (bool): Whether to use distillation. Default: True
    """
    
    def __init__(self, depths: Tuple[int, ...] = (3, 3, 6, 3), 
                 embed_dims: Tuple[int, ...] = (48, 96, 224, 448),
                 mlp_ratios: Tuple[int, ...] = (4, 4, 4, 4),
                 downsamples: Tuple[bool, ...] = (True, True, True, True),
                 in_chans: int = 3, vit_num: int = 0, resolution: int = 224,
                 distillation: bool = True, out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 drop_path_rate: float = 0., **kwargs):
        super().__init__()
        
        self.resolution = resolution
        self.distillation = distillation
        self.vit_num = vit_num
        self.out_indices = out_indices
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                EfficientFormerBlock(
                    dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop_path=drop_path_rate
                ) for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            if downsamples[i] and i < len(depths) - 1:
                downsample = nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=2, padding=1)
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())
        
        # Add ViT blocks if specified
        if vit_num > 0:
            self.vit_blocks = nn.Sequential(*[
                EfficientFormerBlock(
                    dim=embed_dims[-1], mlp_ratio=mlp_ratios[-1], drop_path=drop_path_rate
                ) for _ in range(vit_num)
            ])
        
        # Add norm layers for outputs
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in self.out_indices])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward function."""
        outs = []
        
        for i, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            
            if i in self.out_indices:
                norm_layer = self.norm_layers[self.out_indices.index(i)]
                # Normalize and permute for YOLOv5 neck
                B, C, H, W = x.shape
                out = x.flatten(2).transpose(1, 2)
                out = norm_layer(out)
                out = out.transpose(1, 2).view(B, C, H, W).contiguous()
                outs.append(out)
            
            x = downsample(x)
        
        if self.vit_num > 0:
            x = self.vit_blocks(x)
        
        return tuple(outs)


class EfficientFormerBlock(nn.Module):
    """EfficientFormer block with meta-block design."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4., drop_path: float = 0.):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# YOLOv5 Integration Classes
class SwinTransformerBackbone(nn.Module):
    """Swin Transformer backbone wrapper for YOLOv5 integration."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # Default Swin-T configuration
        default_cfg = {
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': 7,
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.2,
            'ape': False,
            'patch_norm': True,
            'out_indices': (1, 2, 3),  # Output from stages 2, 3, 4 for YOLOv5
            'frozen_stages': -1,
            'use_checkpoint': False
        }
        
        # Update with provided config
        default_cfg.update(cfg)
        
        self.model = SwinTransformer(**default_cfg)
        
        # Channel mapping for YOLOv5 neck
        self.out_channels = [
            int(default_cfg['embed_dim'] * 2 ** i) 
            for i in default_cfg['out_indices']
        ]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model(x)
    
    def init_weights(self, pretrained: Optional[str] = None):
        self.model.init_weights(pretrained)


class EfficientFormerBackbone(nn.Module):
    """EfficientFormer backbone wrapper for YOLOv5 integration."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # Default EfficientFormer-S1 configuration
        default_cfg = {
            'depths': (3, 3, 6, 3),
            'embed_dims': (48, 96, 224, 448),
            'mlp_ratios': (4, 4, 4, 4),
            'downsamples': (True, True, True, True),
            'vit_num': 0,
            'resolution': 224,
            'distillation': True,
            'out_indices': (0, 1, 2, 3),
            'drop_path_rate': 0.
        }
        
        # Update with provided config
        default_cfg.update(cfg)
        
        self.model = EfficientFormer(**default_cfg)
        
        # Channel mapping for YOLOv5 neck
        self.out_channels = [default_cfg['embed_dims'][i] for i in default_cfg['out_indices']]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model(x)


# Factory function for creating transformer backbones
def create_transformer_backbone(backbone_type: str, cfg: dict) -> nn.Module:
    """Create transformer backbone for YOLOv5.
    
    Args:
        backbone_type: Type of transformer backbone ('swin' or 'efficientformer')
        cfg: Configuration dictionary
        
    Returns:
        Transformer backbone module
    """
    backbone_type = backbone_type.lower()
    
    if backbone_type == 'swin':
        return SwinTransformerBackbone(cfg)
    elif backbone_type == 'efficientformer':
        return EfficientFormerBackbone(cfg)
    else:
        raise ValueError(f"Unknown transformer backbone type: {backbone_type}. "
                        f"Supported types: 'swin', 'efficientformer'")


# YOLOv5 model configuration for transformer backbones
TRANSFORMER_BACKBONE_CONFIGS = {
    'swin_tiny': {
        'backbone_type': 'swin',
        'cfg': {
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': 7,
            'out_indices': (1, 2, 3)
        }
    },
    'swin_small': {
        'backbone_type': 'swin',
        'cfg': {
            'embed_dim': 96,
            'depths': (2, 2, 18, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': 7,
            'out_indices': (1, 2, 3)
        }
    },
    'swin_base': {
        'backbone_type': 'swin',
        'cfg': {
            'embed_dim': 128,
            'depths': (2, 2, 18, 2),
            'num_heads': (4, 8, 16, 32),
            'window_size': 7,
            'out_indices': (1, 2, 3)
        }
    },
    'efficientformer_s1': {
        'backbone_type': 'efficientformer',
        'cfg': {
            'depths': (3, 3, 6, 3),
            'embed_dims': (48, 96, 224, 448),
            'mlp_ratios': (4, 4, 4, 4),
            'out_indices': (0, 1, 2, 3)
        }
    },
    'efficientformer_s2': {
        'backbone_type': 'efficientformer',
        'cfg': {
            'depths': (3, 3, 12, 3),
            'embed_dims': (48, 96, 224, 448),
            'mlp_ratios': (4, 4, 4, 4),
            'out_indices': (0, 1, 2, 3)
        }
    },
    'efficientformer_l': {
        'backbone_type': 'efficientformer',
        'cfg': {
            'depths': (3, 3, 18, 3),
            'embed_dims': (64, 128, 320, 512),
            'mlp_ratios': (4, 4, 4, 4),
            'out_indices': (0, 1, 2, 3)
        }
    }
}


# Export compatibility functions
def make_divisible(x: int, divisor: int = 8) -> int:
    """Make x divisible by divisor for export compatibility."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def get_transformer_backbone_channels(backbone_type: str, model_size: str = 'tiny') -> List[int]:
    """Get output channels for transformer backbone.
    
    Args:
        backbone_type: 'swin' or 'efficientformer'
        model_size: Model size variant
        
    Returns:
        List of output channels for each scale
    """
    if backbone_type.lower() == 'swin':
        if model_size == 'tiny':
            return [192, 384, 768]
        elif model_size == 'small':
            return [192, 384, 768]
        elif model_size == 'base':
            return [256, 512, 1024]
        else:
            return [192, 384, 768]
    
    elif backbone_type.lower() == 'efficientformer':
        if model_size == 's1':
            return [96, 224, 448]
        elif model_size == 's2':
            return [96, 224, 448]
        elif model_size == 'l':
            return [128, 320, 512]
        else:
            return [96, 224, 448]
    
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


# Integration with YOLOv5 model parsing
def parse_transformer_backbone(backbone_cfg: dict) -> nn.Module:
    """Parse transformer backbone configuration for YOLOv5.
    
    Args:
        backbone_cfg: Backbone configuration dictionary
        
    Returns:
        Transformer backbone module
    """
    backbone_type = backbone_cfg.get('type', 'swin')
    model_size = backbone_cfg.get('size', 'tiny')
    
    config_key = f"{backbone_type}_{model_size}"
    if config_key in TRANSFORMER_BACKBONE_CONFIGS:
        config = TRANSFORMER_BACKBONE_CONFIGS[config_key]
        return create_transformer_backbone(config['backbone_type'], config['cfg'])
    else:
        # Custom configuration
        cfg = backbone_cfg.get('cfg', {})
        return create_transformer_backbone(backbone_type, cfg)


# ONNX/TensorRT export helpers
class TransformerBackboneWrapper(nn.Module):
    """Wrapper for transformer backbones to ensure export compatibility."""
    
    def __init__(self, backbone: nn.Module, input_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        
        # Pre-compute feature map sizes for different input sizes
        self.feature_sizes = self._compute_feature_sizes()
    
    def _compute_feature_sizes(self) -> List[Tuple[int, int]]:
        """Compute feature map sizes for each backbone stage."""
        sizes = []
        h, w = self.input_size
        
        # Swin Transformer downsampling: 4x (patch_embed) then 2x per stage after first
        # Stage 1: 1/4, Stage 2: 1/8, Stage 3: 1/16, Stage 4: 1/32
        for i in range(3):  # Output 3 scales for YOLOv5
            stride = 8 * (2 ** i)  # 8, 16, 32
            sizes.append((h // stride, w // stride))
        
        return sizes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward with dynamic shape handling for export."""
        features = self.backbone(x)
        
        # Ensure features are in correct order (smallest to largest for YOLOv5 neck)
        if len(features) == 4:
            # Skip the first feature map (1/4 resolution) for YOLOv5
            features = features[1:]
        
        # Reverse order for YOLOv5 neck (P5, P4, P3)
        return tuple(reversed(features))
    
    def get_export_channels(self) -> List[int]:
        """Get output channels for export configuration."""
        if hasattr(self.backbone, 'out_channels'):
            return list(self.backbone.out_channels)
        else:
            # Default channels for common configurations
            return [256, 512, 1024]


# Utility functions for model conversion
def convert_cspdarknet_to_swin(csp_model: nn.Module, swin_model: nn.Module) -> None:
    """Convert CSPDarknet weights to Swin Transformer (for transfer learning)."""
    # This is a simplified conversion - in practice, you'd need more sophisticated mapping
    csp_state = csp_model.state_dict()
    swin_state = swin_model.state_dict()
    
    # Map similar layers where possible
    for name, param in csp_state.items():
        if 'conv.weight' in name and param.shape[0] == swin_state.get('patch_embed.weight', torch.zeros(1)).shape[1]:
            if 'patch_embed.weight' in swin_state:
                # Initialize patch embedding with first conv layer weights
                swin_state['patch_embed.weight'] = param.unsqueeze(-1).unsqueeze(-1)
    
    swin_model.load_state_dict(swin_state, strict=False)


# Example usage and testing
if __name__ == "__main__":
    # Test Swin Transformer backbone
    print("Testing Swin Transformer Backbone...")
    swin_cfg = {
        'embed_dim': 96,
        'depths': (2, 2, 6, 2),
        'num_heads': (3, 6, 12, 24),
        'window_size': 7,
        'out_indices': (1, 2, 3)
    }
    
    swin_backbone = SwinTransformerBackbone(swin_cfg)
    x = torch.randn(2, 3, 640, 640)
    outputs = swin_backbone(x)
    
    print(f"Swin Transformer outputs: {[o.shape for o in outputs]}")
    print(f"Output channels: {swin_backbone.out_channels}")
    
    # Test EfficientFormer backbone
    print("\nTesting EfficientFormer Backbone...")
    eff_cfg = {
        'depths': (3, 3, 6, 3),
        'embed_dims': (48, 96, 224, 448),
        'mlp_ratios': (4, 4, 4, 4),
        'out_indices': (0, 1, 2, 3)
    }
    
    eff_backbone = EfficientFormerBackbone(eff_cfg)
    outputs = eff_backbone(x)
    
    print(f"EfficientFormer outputs: {[o.shape for o in outputs]}")
    print(f"Output channels: {eff_backbone.out_channels}")
    
    # Test export wrapper
    print("\nTesting Export Wrapper...")
    wrapper = TransformerBackboneWrapper(swin_backbone)
    export_outputs = wrapper(x)
    print(f"Export wrapper outputs: {[o.shape for o in export_outputs]}")
    
    # Test factory function
    print("\nTesting Factory Function...")
    for name, config in TRANSFORMER_BACKBONE_CONFIGS.items():
        try:
            backbone = create_transformer_backbone(config['backbone_type'], config['cfg'])
            test_out = backbone(x)
            print(f"{name}: Success - {[o.shape for o in test_out]}")
        except Exception as e:
            print(f"{name}: Failed - {e}")
    
    print("\nAll tests completed!")