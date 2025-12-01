import torch
import torch.nn as nn
import timm
import os
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


class EVA02_STCMT(nn.Module):
    """
    基于 EVA-02 Backbone 的 ST-CMT 网络架构
    """

    def __init__(self, num_classes, camera_num, cfg):
        super(EVA02_STCMT, self).__init__()

        # 1. 自动推断并构建 Backbone (EVA-02)
        # 这一步非常关键，必须匹配权重文件的结构
        model_name = self._get_model_name(cfg.MODEL.PRETRAIN_PATH)
        print(f"Building Backbone: {model_name}")

        self.backbone = timm.create_model(model_name, pretrained=False, img_size=cfg.INPUT.SIZE_TRAIN, num_classes=0, dynamic_img_size=True)

        # 加载你指定的权重
        self._load_pretrained(cfg.MODEL.PRETRAIN_PATH)

        self.in_planes = self.backbone.num_features
        self.num_prefix_tokens = self.backbone.num_prefix_tokens if hasattr(self.backbone, "num_prefix_tokens") else 1
        print(f"Backbone initialized. Dim: {self.in_planes}, Prefix Tokens: {self.num_prefix_tokens}")

        # 2. 构建 ST-CMT 模块
        self.use_cmt = cfg.MODEL.USE_CMT
        if self.use_cmt:
            print(f"Initializing ST-CMT Module (Parts={cfg.MODEL.CMT_NUM_PARTS})...")
            self.cmt_head = CMTModule(in_dim=self.in_planes, num_parts=cfg.MODEL.CMT_NUM_PARTS, num_classes=num_classes)

        # 3. 构建 Baseline 头部
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

        self.train_pair = False

    def train_with_pair(self):
        self.train_pair = True

    def train_with_single(self):
        self.train_pair = False

    def _get_model_name(self, path):
        """
        根据权重路径自动匹配 timm 中的模型名称
        """
        # 你的权重是: eva02_base_patch14_224.mim_in22k.bin
        if "eva02" in path:
            if "large" in path:
                if "448" in path:
                    return "eva02_large_patch14_448.mim_in22k_ft_in1k"
                else:
                    return "eva02_large_patch14_224.mim_in22k"  # 假设有224版本
            else:  # base
                if "448" in path:
                    return "eva02_base_patch14_448.mim_in22k_ft_in1k"
                else:
                    # === 修正点：匹配 224 版本 ===
                    return "eva02_base_patch14_224.mim_in22k"

        # 默认回退
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

        # === 关键：过滤掉不匹配的头部权重 ===
        # EVA02 预训练权重可能包含 head.weight/bias，而我们的 backbone 没有 head
        for k in list(state_dict.keys()):
            if "head" in k:
                del state_dict[k]

        # 处理位置编码插值 (如果有必要)
        if "pos_embed" in state_dict:
            ckpt_pos_shape = state_dict["pos_embed"].shape
            model_pos_shape = self.backbone.pos_embed.shape
            if ckpt_pos_shape != model_pos_shape:
                print(f"Resizing pos_embed: {ckpt_pos_shape} -> {model_pos_shape}")
                state_dict["pos_embed"] = resample_abs_pos_embed(
                    state_dict["pos_embed"], new_size=self.backbone.patch_embed.grid_size, num_prefix_tokens=self.backbone.num_prefix_tokens
                )

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Weight loading result: {msg}")

    def forward(self, x, label=None, cam_label=None, img_wh=None):
        # 1. 提取特征
        features = self.backbone.forward_features(x)

        # EVA-02 的 features 结构是 [B, N+1, D] (CLS在0位)
        global_feat = features[:, 0]
        patch_tokens = features[:, self.num_prefix_tokens :]

        # =====================================================
        # ST-CMT 逻辑
        # =====================================================
        if self.use_cmt:
            # 传入 cam_label
            cmt_out = self.cmt_head(patch_tokens, cam_label)

            if self.training:
                # 解包 5 个值 (F_final, F_comp, Logits, Saliency, AttnMap)
                f_final, f_comp, cls_scores, saliency_scores, attn_map = cmt_out

                # 准备 Triplet Loss
                part_feats_list = [f_final[:, i, :] for i in range(f_final.size(1))]

                return cls_scores, part_feats_list, f_comp, saliency_scores, attn_map
            else:
                return cmt_out[0]

        # =====================================================
        # Baseline 逻辑
        # =====================================================
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
