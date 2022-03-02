# -----------------
# The post layer specifically designed for imagenet classification follows VOLO.
# Reference: https://github.com/sail-sg/volo/blob/main/models/volo.py
# -----------------

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import numpy as np
from .lvt import lvt

__all__ = ['lvt_cls']

def _cfg(**kwargs):
    return {
        'num_classes': 1000,
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs,
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim,
                            self.head_dim * self.num_heads * 2,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(
            B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed

class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

def get_block(block_type, **kargs):
    if block_type == 'ca':
        return ClassBlock(**kargs)

def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class lite_vision_transformer_with_post_layers(lvt):
    def __init__(self, post_layers, **kwargs):
        super().__init__(**kwargs)
        self.post_network = nn.ModuleList([
            get_block(
                post_layers[i],
                dim=kwargs['embed_dims'][-1],
                num_heads=kwargs['num_heads'][-1],
                mlp_ratio=kwargs['mlp_ratios'][-1],
                qkv_bias=kwargs.pop('qkv_bias', False),
                qk_scale=kwargs.pop('qk_scale', None),
                attn_drop=kwargs.pop('attn_drop_rate', 0.),
                drop_path=0.,
                norm_layer=kwargs.pop('norm_layer', nn.LayerNorm))
            for i in range(len(post_layers))])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[-1]))
        trunc_normal_(self.cls_token, std=.02)

        # Enable token labelling.
        self.pooling_scale = 8
        self.beta = 1.0
        self.aux_head = nn.Linear(
            self.embed_dims[-1],
            self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.head = nn.Linear(
            self.embed_dims[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward(self, x):
        x = self.backbone[0][0](x)
        
        if self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1,sbby1,sbbx2,sbby2=self.pooling_scale*bbx1,self.pooling_scale*bby1,\
                                    self.pooling_scale*bbx2,self.pooling_scale*bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        
        x = self.backbone[0][1](x)
        
        for idx in range(1, len(self.backbone)):
            x = self.backbone[idx](x)
        
        B,_,_,C = x.shape
        x = x.reshape(B,-1,C)
        x = self.forward_cls(x) # cls_token
        x = self.norm(x)
        
        x_cls = self.head(x[:, 0])
        
        x_aux = self.aux_head(
            x[:, 1:]
        )  # generate classes in all feature tokens, see token labeling
        
        if self.training:  # reverse "mix token", see token labeling for details.
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])
            # return these: 1. class token, 2. classes from all feature tokens, 3. bounding box
            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        else:
            return x_cls + 0.5 * x_aux.max(1)[0]

@register_model
class lvt_cls(lite_vision_transformer_with_post_layers):
    def __init__(self, rasa_cfg=None, with_cls_head=True, **kwargs):
        super().__init__(
            layers=[2,2,2,2],
            patch_size=4,
            embed_dims=[64,64,160,256],
            num_heads=[2,2,5,8],
            mlp_ratios=[4,8,4,4],
            mlp_depconv=[False, True, True, True],
            sr_ratios=[8,4,2,1],
            sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
            rasa_cfg=rasa_cfg,
            with_cls_head=with_cls_head,
            post_layers=['ca', 'ca'], # post_layers
        )
        self.default_cfg = _cfg(crop_pct=0.96)
