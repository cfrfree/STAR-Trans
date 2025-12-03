import torch
import torch.nn as nn
import timm
import os
import copy
import logging

# 引入 ST-CMT 模块
from .backbones.cmt_module import CMTModule
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from timm.layers import resample_abs_pos_embed


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# === 双模态 Patch Embedding 模块 ===
class DualModalPatchEmbed(nn.Module):
    def __init__(self, original_embed):
        super().__init__()
        self.rgb_embed = original_embed
        self.sar_embed = copy.deepcopy(original_embed)
        self.current_cam_label = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.rgb_embed, name)

    def forward(self, x):
        if self.current_cam_label is None:
            return self.rgb_embed(x)

        cam_label = self.current_cam_label
        B = x.shape[0]
        rgb_idx = cam_label == 0
        sar_idx = cam_label == 1

        x_rgb = None
        x_sar = None

        if rgb_idx.any():
            x_rgb = self.rgb_embed(x[rgb_idx])
        if sar_idx.any():
            x_sar = self.sar_embed(x[sar_idx])

        if x_rgb is not None and x_sar is None:
            return x_rgb
        if x_sar is not None and x_rgb is None:
            return x_sar

        out_shape = list(x_rgb.shape)
        out_shape[0] = B
        out = torch.zeros(out_shape, dtype=x_rgb.dtype, device=x_rgb.device)
        out[rgb_idx] = x_rgb
        out[sar_idx] = x_sar
        return out


class EVA02_STCMT(nn.Module):
    def __init__(self, num_classes, camera_num, cfg):
        super(EVA02_STCMT, self).__init__()

        # 1. 构建 Backbone
        model_name = self._get_model_name(cfg.MODEL.PRETRAIN_PATH)
        print(f"Building Backbone: {model_name}")

        self.backbone = timm.create_model(model_name, pretrained=False, img_size=cfg.INPUT.SIZE_TRAIN, num_classes=0, dynamic_img_size=True)

        self._load_pretrained(cfg.MODEL.PRETRAIN_PATH)

        # 双流 Tokenizer 开关
        self.use_dual_tokenizer = getattr(cfg.MODEL, "DUAL_TOKENIZER", True)
        if self.use_dual_tokenizer:
            print(">>> [Config] DUAL_TOKENIZER is ON.")
            self.backbone.patch_embed = DualModalPatchEmbed(self.backbone.patch_embed)
        else:
            print(">>> [Config] DUAL_TOKENIZER is OFF.")

        self.in_planes = self.backbone.num_features
        self.num_prefix_tokens = self.backbone.num_prefix_tokens if hasattr(self.backbone, "num_prefix_tokens") else 1
        print(f"Backbone initialized. Dim: {self.in_planes}")

        # 2. ST-CMT 模块
        self.use_cmt = cfg.MODEL.USE_CMT
        if self.use_cmt:
            print(f"Initializing ST-CMT Module (Parts={cfg.MODEL.CMT_NUM_PARTS})...")
            self.cmt_head = CMTModule(in_dim=self.in_planes, num_parts=cfg.MODEL.CMT_NUM_PARTS, num_classes=num_classes)

        # 3. Baseline 头部
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        if self.ID_LOSS_TYPE == "arcface":
            self.classifier = Arcface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "cosface":
            self.classifier = Cosface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "amsoftmax":
            self.classifier = AMSoftmax(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "circle":
            self.classifier = CircleLoss(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # === 预训练模式相关参数 ===
        self.train_pair = False
        # 初始 logit_scale (log(1/0.07) ≈ 2.6592)，用于对比学习
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def train_with_pair(self):
        self.train_pair = True

    def train_with_single(self):
        self.train_pair = False

    def _get_model_name(self, path):
        if "large" in path:
            return "eva02_large_patch14_448.mim_in22k_ft_in1k"
        else:
            return "eva02_base_patch14_224.mim_in22k"

    def _load_pretrained(self, path):
        if not os.path.isfile(path):
            print(f"Warning: Pretrained path {path} not found. Using random init.")
            return
        print(f"Loading pretrained weights from: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        for k in list(state_dict.keys()):
            if "head" in k:
                del state_dict[k]

        if "pos_embed" in state_dict:
            state_dict["pos_embed"] = resample_abs_pos_embed(
                state_dict["pos_embed"], new_size=self.backbone.patch_embed.grid_size, num_prefix_tokens=self.backbone.num_prefix_tokens
            )
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x, label=None, cam_label=None, img_wh=None):
        # 注入模态标签 (用于 Dual Tokenizer)
        if self.use_dual_tokenizer and hasattr(self.backbone.patch_embed, "current_cam_label"):
            self.backbone.patch_embed.current_cam_label = cam_label

        # 1. 提取特征
        features = self.backbone.forward_features(x)

        global_feat = features[:, 0]
        patch_tokens = features[:, self.num_prefix_tokens :]

        # === 预训练模式 (Contrastive Learning) ===
        # 此时 batch 包含 [RGBs; SARs]，我们需要计算它们之间的相似度
        if self.train_pair:
            # 假设前半部分是 RGB，后半部分是 SAR (由 Dataloader 保证)
            b_s = global_feat.size(0)
            opt_embeds = global_feat[0 : b_s // 2]
            sar_embeds = global_feat[b_s // 2 :]

            # 归一化
            opt_embeds = opt_embeds / opt_embeds.norm(p=2, dim=-1, keepdim=True)
            sar_embeds = sar_embeds / sar_embeds.norm(p=2, dim=-1, keepdim=True)

            # 计算 Logits
            logit_scale = self.logit_scale.exp()
            logits_per_sar = torch.matmul(sar_embeds, opt_embeds.t()) * logit_scale

            return logits_per_sar

        # === ST-CMT 逻辑 (微调模式) ===
        if self.use_cmt:
            cmt_out = self.cmt_head(patch_tokens, cam_label)
            if self.training:
                # 解包 5 个值
                f_final, f_comp, cls_scores, saliency_scores, attn_map = cmt_out
                part_feats_list = [f_final[:, i, :] for i in range(f_final.size(1))]
                return cls_scores, part_feats_list, f_comp, saliency_scores, attn_map
            else:
                return cmt_out[0]

        # === Baseline 逻辑 ===
        feat = self.bottleneck(global_feat)
        if self.training:
            if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        new_state_dict = {}
        for k, v in param_dict.items():
            if k.startswith("module."):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)


def make_model(cfg, num_class, camera_num):
    model = EVA02_STCMT(num_class, camera_num, cfg)
    return model
