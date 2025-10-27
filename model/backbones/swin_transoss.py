import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from timm.models.swin_transformer import PatchEmbed as SwinPatchEmbed
from timm.models.layers import trunc_normal_
from .vit_transoss import WHPatchEmbedding
from functools import partial


class SwinTransOSS(nn.Module):
    """
    Swin-Transformer-based TransOSS model @ 224x224.
    """

    def __init__(
        self,
        cfg,
        timm_name,  # e.g., 'swin_base_patch4_window7_224'
        camera_num=2,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        patch_size=4,
        **kwargs,
    ):
        super(SwinTransOSS, self).__init__()
        self.cfg = cfg
        self.camera_num = camera_num

        # 1. 创建基础的Swin Transformer模型
        # (双重保险: 同时提供 timm_name 和 显式参数)
        self.base = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            img_size=cfg.INPUT.SIZE_TRAIN,  # 应该是 [224, 224]
            # (核心修改) 强制覆盖架构参数
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
            # 传递其他配置
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            num_classes=0,  # 禁用 timm 的 head
        )

        self.embed_dim = self.base.embed_dim
        self.in_planes = self.embed_dim

        # 224x224 (patch 4x4) -> 56x56 = 3136 个 patches
        self.num_patches = self.base.patch_embed.num_patches

        # 2. 创建双模态PatchEmbed (光学, SAR)
        self.patch_embed = self.base.patch_embed
        self.patch_embed_SAR = SwinPatchEmbed(
            img_size=cfg.INPUT.SIZE_TRAIN,
            patch_size=self.base.patch_embed.patch_size[0],
            in_chans=3,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm,
        )

        # 3. 添加ViT风格的[CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # 4. 替换Swin-T的绝对位置嵌入为ViT风格的可学习位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        if hasattr(self.base, "absolute_pos_embed"):
            self.base.absolute_pos_embed = None

        # 5. 添加模态信息嵌入 (MIE)
        self.mie_coe = cfg.MODEL.MIE_COE
        if self.camera_num > 0 and cfg.MODEL.MIE:
            self.mie_embed = nn.Parameter(torch.zeros(self.camera_num, 1, self.embed_dim))
            trunc_normal_(self.mie_embed, std=0.02)

        # 6. 添加舰船尺寸嵌入 (SSE)
        self.sse = cfg.MODEL.SSE
        if self.sse:
            self.wh_embed = WHPatchEmbedding(3, self.embed_dim)
            self._init_weights(self.wh_embed.linear_layer)

        # 7. 原生定义 pos_drop 和 norm
        self.pos_drop = nn.Dropout(p=cfg.MODEL.DROP_OUT)
        self.layers = self.base.layers
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.embed_dim)

        # 8. 禁用 base 模型中我们已经替换或原生定义的组件
        self.base.avgpool = nn.Identity()
        self.base.head = nn.Identity()
        self.base.norm = nn.Identity()
        self.base.pos_drop = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, camera_id, img_wh):
        B = x.shape[0]

        # 1. 双模态 Patch Embedding
        rgb_id = torch.where(camera_id == 0)[0]
        sar_id = torch.where(camera_id == 1)[0]

        patches = torch.zeros(B, self.num_patches, self.embed_dim, device=x.device, dtype=x.dtype)

        if len(rgb_id) > 0:
            patches[rgb_id] = self.patch_embed(x[rgb_id])
        if len(sar_id) > 0:
            patches[sar_id] = self.patch_embed_SAR(x[sar_id])

        x = patches

        # 2. 添加 [CLS] Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 添加 Positional Embedding (ViT-style)
        x = x + self.pos_embed

        # 4. 添加 Modality Information Embedding (MIE)
        if hasattr(self, "mie_embed"):
            x = x + self.mie_coe * self.mie_embed[camera_id]

        # 5. 添加 Ship Size Embedding (SSE)
        if self.sse and hasattr(self, "wh_embed"):
            wh_tokens = self.wh_embed(img_wh).unsqueeze(1)
            x = torch.cat((x, wh_tokens), dim=1)

        # 6. 应用 pos_drop
        x = self.pos_drop(x)

        # 7. 通过Swin Transformer的各个Stage
        x = self.layers(x)

        # 8. 最终的LayerNorm
        x = self.norm(x)

        # 9. 返回[CLS] Token的特征
        return x[:, 0]

    def forward(self, x, cam_label=None, img_wh=None):
        return self.forward_features(x, cam_label, img_wh)

    def load_param(self, model_path):
        """
        加载预训练权重 @ 224x224
        """
        param_dict = torch.load(model_path, map_location="cpu")
        if "model" in param_dict:
            param_dict = param_dict["model"]
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        # 1. 清理 'module.' 前缀
        new_state_dict = {}
        for k, v in param_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        param_dict = new_state_dict

        # 2. 处理 Positional Embedding (绝对位置嵌入)
        if "absolute_pos_embed" in param_dict:
            orig_pos_embed = param_dict["absolute_pos_embed"]
            if orig_pos_embed.shape[1] == self.num_patches:
                print("Loading absolute_pos_embed and grafting [CLS] token.")
                with torch.no_grad():
                    self.pos_embed[:, 1:, :] = orig_pos_embed
            else:
                print(f"ERROR: PosEmbed mismatch. CKPT: {orig_pos_embed.shape}, Model: {self.num_patches}")
            del param_dict["absolute_pos_embed"]

        # 3. 删除预训练的 head (分类头)
        param_dict.pop("head.weight", None)
        param_dict.pop("head.bias", None)

        # 4. 加载所有其他权重
        # (因为架构是强制匹配的, 现在 Downsampler 应该匹配了)
        msg = self.base.load_state_dict(param_dict, strict=False)
        print(f"Loaded base Swin-T weights from {model_path}.")

        missing_keys = [k for k in msg.missing_keys if "absolute_pos" not in k]
        unexpected_keys = [k for k in msg.unexpected_keys if "absolute_pos" not in k]

        if missing_keys:
            print(f"Missing keys (filtered): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (filtered): {unexpected_keys}")

        # 5. 初始化SAR的patch_embed (从光学的patch_embed复制)
        if "patch_embed.proj.weight" in param_dict:
            print("Initializing patch_embed_SAR from patch_embed.")
            with torch.no_grad():
                self.patch_embed_SAR.proj.weight.copy_(param_dict["patch_embed.proj.weight"])
                self.patch_embed_SAR.proj.bias.copy_(param_dict["patch_embed.proj.bias"])
                self.patch_embed_SAR.norm.weight.copy_(param_dict["patch_embed.norm.weight"])
                self.patch_embed_SAR.norm.bias.copy_(param_dict["patch_embed.norm.bias"])

        # 6. 初始化我们自己的 norm 层
        if "norm.weight" in param_dict:
            print("Initializing native 'norm' layer from base 'norm' layer.")
            with torch.no_grad():
                self.norm.weight.copy_(param_dict["norm.weight"])
                self.norm.bias.copy_(param_dict["norm.bias"])


# =======================================================
#               模型工厂函数
# =======================================================


def swin_tiny_patch4_224_TransOSS(cfg, camera_num, **kwargs):
    """
    Swin-Tiny TransOSS 模型
    """
    timm_name = "swin_tiny_patch4_window7_224"

    # (核心修改) 显式定义Tiny的架构
    model_args = dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        patch_size=4,
        **kwargs,
    )

    model = SwinTransOSS(cfg, timm_name, camera_num, **model_args)
    return model


def swin_base_patch4_224_TransOSS(cfg, camera_num, **kwargs):
    """
    Swin-Base TransOSS 模型
    """
    timm_name = "swin_base_patch4_window7_224"

    # (核心修改) 显式定义Base的架构
    model_args = dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        patch_size=4,
        **kwargs,
    )

    model = SwinTransOSS(cfg, timm_name, camera_num, **model_args)
    return model
