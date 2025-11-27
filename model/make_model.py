import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.cmt_module import CMTModule
import timm

from .backbones.vit_transoss import vit_base_patch16_224_TransOSS, vit_large_patch16_224_TransOSS

from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


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


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == "resnet50":
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            print("using resnet50 as a backbone")
        else:
            print("unsupported backbone! but got {}".format(model_name))

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained model......from {}".format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == "after":
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory, logit_scale_init_value=2.6592):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_type = cfg.MODEL.TRANSFORMER_TYPE

        print("using Transformer_type: {} as a backbone".format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.MIE:
            camera_num = camera_num
        else:
            camera_num = 0

        # 2. --- 修改模型构建逻辑 ---
        if cfg.MODEL.TRANSFORMER_TYPE not in factory:
            raise ValueError("Unsupported model type: {}".format(cfg.MODEL.TRANSFORMER_TYPE))

        if "vit" in cfg.MODEL.TRANSFORMER_TYPE:
            # ViT-specific arguments
            print("Building ViT-TransOSS model...")
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                mie_coe=cfg.MODEL.MIE_COE,
                camera=camera_num,
                stride_size=cfg.MODEL.STRIDE_SIZE,  # ViT 需要 stride_size
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                sse=cfg.MODEL.SSE,
            )
        else:
            raise ValueError("Unknown Transformer type logic")

        self.in_planes = self.base.embed_dim
        # === 新增 ===
        self.use_cmt = cfg.MODEL.USE_CMT
        if self.use_cmt:
            # 假设 patch_size=16, 图像高 256 -> h_num = 16
            # 建议将 num_parts 设为 4 或 8
            self.cmt_head = CMTModule(in_dim=self.in_planes, num_parts=cfg.MODEL.CMT_NUM_PARTS, num_classes=num_classes)

            # 计算 ViT 输出的 H 和 W 的 token 数量
            self.h_num = cfg.INPUT.SIZE_TRAIN[0] // 16
            self.w_num = cfg.INPUT.SIZE_TRAIN[1] // 16
        # ===========
        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained model......from {}".format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == "arcface":
            print("using {} with s:{}, m: {}".format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "cosface":
            print("using {} with s:{}, m: {}".format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "amsoftmax":
            print("using {} with s:{}, m: {}".format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == "circle":
            print("using {} with s:{}, m: {}".format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.train_pair = False
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def train_with_pair(
        self,
    ):
        self.train_pair = True

    def train_with_single(
        self,
    ):
        self.train_pair = False

    def forward(self, x, label=None, cam_label=None, img_wh=None):
        # 1. 获取 Backbone 输出
        # 注意：现在 backbone 返回的是 (cls_token, patch_tokens)
        outputs = self.base(x, cam_label=cam_label, img_wh=img_wh)

        # 2. 关键修复：正确解包并定义 global_feat
        if isinstance(outputs, tuple):
            global_feat, patch_tokens = outputs
        else:
            # 兼容旧逻辑（如果 backbone 没改）
            global_feat = outputs
            patch_tokens = None

        # ---------------------------------------------------
        # 3. CMT 逻辑分支
        # ---------------------------------------------------
        if self.use_cmt and patch_tokens is not None:
            # === 修改处 1：接收 4 个返回值 ===
            f_final, f_comp, cls_scores, saliency_scores = self.cmt_head(patch_tokens, cam_label, self.h_num, self.w_num)

            if self.training:
                # 将 f_final 拆分为 list 传给 loss
                part_feats_list = [f_final[:, i, :] for i in range(f_final.size(1))]

                # === 修改处 2：将 saliency_scores 也返回出去 ===
                return cls_scores, part_feats_list, f_comp, saliency_scores
            else:
                # 测试阶段返回拼接特征 (测试阶段 cmt_head 只返回 1 个有效值，其他是 None，需要对应处理)
                # 注意：CMTModule 测试阶段返回的是 (F_final_view, None, None, None)
                # 所以这里直接返回第一个即可
                return f_final  # f_final 已经在 CMTModule 里 view 好了

        # ---------------------------------------------------
        # 4. 原有逻辑分支 (当不使用 CMT 时，作为兜底)
        # ---------------------------------------------------
        # 只有定义了 global_feat，下面的代码才不会报错
        if self.training:
            if self.train_pair:
                # 对比学习逻辑 (保持原样)
                b_s = global_feat.size(0)
                opt_embeds = global_feat[0 : b_s // 2]
                sar_embeds = global_feat[b_s // 2 :]
                opt_embeds = opt_embeds / opt_embeds.norm(p=2, dim=-1, keepdim=True)
                sar_embeds = sar_embeds / sar_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_sar = torch.matmul(sar_embeds, opt_embeds.t()) * logit_scale
                return logits_per_sar
            else:
                # 报错的行就在这里，现在 global_feat 有定义了，就不会报错了
                feat = self.bottleneck(global_feat)
                if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat
        else:
            if self.neck_feat == "after":
                feat = self.bottleneck(global_feat)
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        """
        加载权重的鲁棒版本，能自动处理 'module.' 前缀。
        """
        param_dict = torch.load(trained_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        # 创建一个新的 state_dict 来存储处理过的键
        new_state_dict = {}
        for k, v in param_dict.items():
            if k.startswith("module."):
                # 去掉 'module.' 前缀
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict)
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


__factory_T_type = {
    "vit_base_patch16_224_TransOSS": vit_base_patch16_224_TransOSS,
    "vit_large_patch16_224_TransOSS": vit_large_patch16_224_TransOSS,
}


# def make_model(cfg, num_class, camera_num):
#     if cfg.MODEL.NAME == "transformer":
#         # 4. --- 修改这里的逻辑 ---
#         if cfg.MODEL.TRANSFORMER_TYPE in __factory_T_type:
#             model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
#         else:
#             raise ValueError(f"Unsupported Transformer type: {cfg.MODEL.TRANSFORMER_TYPE}")
#     else:
#         model = Backbone(num_class, cfg)
#     return model

import timm
import torch.nn as nn

# ... (保留原有的引用)


def make_model(cfg, num_class, camera_num):
    # 如果检测到配置的是 transformer
    if cfg.MODEL.NAME == "transformer":
        # 判断是否是要用 DINOv3 这种特殊权重的
        if "dinov3" in cfg.MODEL.PRETRAIN_PATH or "rope" in cfg.MODEL.TRANSFORMER_TYPE:
            print(f"检测到 DINOv3/RoPE 权重，正在使用 timm 构建模型...")

            # 1. 使用 timm 创建兼容 DINOv3 的模型结构
            # 注意：这里的模型名 'vit_base_patch16_224.dinov3_lvd1689m' 是 timm 中对应的名称
            # 如果你的 timm 版本较旧，可能需要更新: pip install --upgrade timm
            model = timm.create_model(
                "vit_base_patch16_224.dinov3_lvd1689m",  # 指定架构
                pretrained=False,  # 我们手动加载权重
                img_size=cfg.INPUT.SIZE_TRAIN,  # [256, 128]
                num_classes=0,  # 移除分类头，只做特征提取
            )

            # 2. 加载你下载的权重
            print(f"正在加载权重: {cfg.MODEL.PRETRAIN_PATH}")
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

            # 处理 timm 可能的 key 不匹配 (例如 norm.weight)
            # 通常 timm 的权重和模型定义是匹配的，直接加载即可
            msg = model.load_state_dict(checkpoint, strict=False)
            print(f"权重加载结果: {msg}")

            # 3. 封装一下以适配你的 ReID 训练代码接口
            # 你的代码期望 model(x) 输出 (cls_score, global_feat)
            class ReIDWrapper(nn.Module):
                def __init__(self, backbone, input_dim, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.in_planes = input_dim

                    # 定义你的分类头 (BNNeck Head)
                    self.bottleneck = nn.BatchNorm1d(self.in_planes)
                    self.bottleneck.bias.requires_grad_(False)
                    self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)

                    # 初始化
                    self.bottleneck.apply(weights_init_kaiming)
                    self.classifier.apply(weights_init_classifier)

                def forward(self, x, label=None, cam_label=None, img_wh=None):
                    # Backbone 提取特征 (B, C, H, W) 或 (B, N, C)
                    # DINOv3 forward_features 输出通常是 (B, N, C)
                    features = self.backbone.forward_features(x)

                    # 取 CLS token (第一个 token)
                    # 或者如果是 patch only，可能需要 average pool
                    global_feat = features[:, 0]

                    feat = self.bottleneck(global_feat)

                    if self.training:
                        cls_score = self.classifier(feat)
                        return cls_score, global_feat
                    else:
                        return feat  # 测试阶段返回 BN 后的特征

            # 包装模型
            reid_model = ReIDWrapper(model, input_dim=768, num_classes=num_class)
            return reid_model

        # ... (保留原有的 else 逻辑处理普通 ViT)
        elif cfg.MODEL.TRANSFORMER_TYPE in __factory_T_type:
            model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
        else:
            raise ValueError(f"Unsupported Transformer type: {cfg.MODEL.TRANSFORMER_TYPE}")
    else:
        model = Backbone(num_class, cfg)
    return model
