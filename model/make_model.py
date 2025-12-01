import torch
import torch.nn as nn
import timm
import os
from timm.layers import resample_abs_pos_embed

from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_transoss import vit_base_patch16_224_TransOSS, vit_large_patch16_224_TransOSS
from .backbones.cmt_module import CMTModule
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
        else:
            print("unsupported backbone! but got {}".format(model_name))

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)

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

    def train_with_pair(self):
        pass

    def train_with_single(self):
        pass


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory, logit_scale_init_value=2.6592):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_type = cfg.MODEL.TRANSFORMER_TYPE

        if cfg.MODEL.MIE:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.TRANSFORMER_TYPE not in factory:
            raise ValueError("Unsupported model type: {}".format(cfg.MODEL.TRANSFORMER_TYPE))

        if "vit" in cfg.MODEL.TRANSFORMER_TYPE:
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                mie_coe=cfg.MODEL.MIE_COE,
                camera=camera_num,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                sse=cfg.MODEL.SSE,
            )

        self.in_planes = self.base.embed_dim
        self.use_cmt = cfg.MODEL.USE_CMT
        if self.use_cmt:
            self.cmt_head = CMTModule(in_dim=self.in_planes, num_parts=cfg.MODEL.CMT_NUM_PARTS, num_classes=num_classes)
            self.h_num = cfg.INPUT.SIZE_TRAIN[0] // 16
            self.w_num = cfg.INPUT.SIZE_TRAIN[1] // 16
        self.use_deen = cfg.MODEL.USE_DEEN
        if self.use_deen:
            print("Building Diverse Embedding Generator (UFE)...")
            self.deen_head = DiverseEmbeddingGenerator(in_dim=self.in_planes, num_variations=3)

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)

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

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.train_pair = False
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def train_with_pair(self):
        self.train_pair = True

    def train_with_single(self):
        self.train_pair = False

    def forward(self, x, label=None, cam_label=None, img_wh=None):
        outputs = self.base(x, cam_label=cam_label, img_wh=img_wh)
        if isinstance(outputs, tuple):
            global_feat, patch_tokens = outputs
        else:
            global_feat = outputs
            patch_tokens = None

        if self.training:
            f_final, f_comp, cls_scores, saliency_scores, attn_map = cmt_out

            # Step B: UFE (特征扩张) - 仅在 ST-CMT 之后进行
            f_generated = None
            if self.use_deen:  # 建议改名为 self.use_ufe 或保持 use_deen
                # 输入: 校准后的 f_final [B, P, D]
                # 输出: 扩张的变体 [B, K, P, D]
                f_generated = self.deen_head(f_final)

            # Step C: 打包返回 (共 6 个元素)
            # 1. cls_scores (用于 ID Loss)
            # 2. part_feats_list (用于 Triplet Loss)
            # 3. f_comp (用于 Cyc Loss)
            # 4. saliency_scores (用于 Cyc Loss 加权)
            # 5. attn_map (用于 Topo Loss)
            # 6. f_generated (用于 CPM Loss)

            part_feats_list = [f_final[:, i, :] for i in range(f_final.size(1))]

            return cls_scores, part_feats_list, f_comp, saliency_scores, attn_map, f_generated
        else:
            # 测试阶段: 只用 ST-CMT 对齐后的特征
            f_final = cmt_out[0]
            return f_final  # [B, P*D]

        if self.training:
            if self.train_pair:
                b_s = global_feat.size(0)
                opt_embeds = global_feat[0 : b_s // 2]
                sar_embeds = global_feat[b_s // 2 :]
                opt_embeds = opt_embeds / opt_embeds.norm(p=2, dim=-1, keepdim=True)
                sar_embeds = sar_embeds / sar_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_sar = torch.matmul(sar_embeds, opt_embeds.t()) * logit_scale
                return logits_per_sar
            else:
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


__factory_T_type = {
    "vit_base_patch16_224_TransOSS": vit_base_patch16_224_TransOSS,
    "vit_large_patch16_224_TransOSS": vit_large_patch16_224_TransOSS,
}


# =====================================================================
#  核心工具：权重映射
# =====================================================================
def remap_dinov3_checkpoint(checkpoint):
    new_state_dict = {}
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    for k, v in checkpoint.items():
        if "storage_tokens" in k:
            k = k.replace("storage_tokens", "reg_token")
        if "ls1.gamma" in k:
            k = k.replace("ls1.gamma", "gamma_1")
        if "ls2.gamma" in k:
            k = k.replace("ls2.gamma", "gamma_2")
        if "mask_token" in k or "rope_embed.periods" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict


def remap_pe_checkpoint(checkpoint):
    """
    将 PE-Core (OpenCLIP风格) 的权重映射到 timm ViT 格式
    """
    print(">>> 正在执行 PE/CLIP 权重映射逻辑 (CLIP -> timm ViT)...")
    new_state_dict = {}
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    mapped_count = 0
    for k, v in state_dict.items():
        # 1. 过滤掉文本编码器，只保留视觉部分
        if not k.startswith("visual."):
            continue

        # 2. 去掉前缀
        new_k = k.replace("visual.", "")

        # 3. 映射层名称
        if new_k == "class_embedding":
            new_k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(0)  # (D,) -> (1, 1, D)
        elif new_k == "positional_embedding":
            new_k = "pos_embed"
            v = v.unsqueeze(0)  # (L, D) -> (1, L, D)
        elif new_k == "conv1.weight":
            new_k = "patch_embed.proj.weight"
        elif new_k == "ln_post.weight":
            new_k = "norm.weight"
        elif new_k == "ln_post.bias":
            new_k = "norm.bias"
        elif "transformer.resblocks" in new_k:
            new_k = new_k.replace("transformer.resblocks.", "blocks.")
            if "ln_1" in new_k:
                new_k = new_k.replace("ln_1", "norm1")
            elif "ln_2" in new_k:
                new_k = new_k.replace("ln_2", "norm2")
            elif "attn.in_proj_weight" in new_k:
                new_k = new_k.replace("attn.in_proj_weight", "attn.qkv.weight")
            elif "attn.in_proj_bias" in new_k:
                new_k = new_k.replace("attn.in_proj_bias", "attn.qkv.bias")
            elif "attn.out_proj" in new_k:
                new_k = new_k.replace("attn.out_proj", "attn.proj")
            elif "mlp.c_fc" in new_k:
                new_k = new_k.replace("mlp.c_fc", "mlp.fc1")
            elif "mlp.c_proj" in new_k:
                new_k = new_k.replace("mlp.c_proj", "mlp.fc2")

        if "proj" in new_k and "patch_embed" not in new_k and "attn" not in new_k and "mlp" not in new_k:
            continue

        new_state_dict[new_k] = v
        mapped_count += 1

    print(f">>> 映射完成: 提取了 {mapped_count} 个视觉层权重")
    return new_state_dict


def make_model(cfg, num_class, camera_num):
    if cfg.MODEL.NAME == "transformer":
        path = cfg.MODEL.PRETRAIN_PATH
        print(f"MakeModel 收到权重路径: {path}")

        # === 判定是否使用 timm ===
        is_timm_model = False
        # 只要不是我们的旧代码名称，或者包含了 timm 支持的关键词
        if "dinov3" in path or "eva02" in path or "rope" in cfg.MODEL.TRANSFORMER_TYPE:
            is_timm_model = True
        elif "PE" in path.lower() or "core" in path.lower():  # 支持 PE-Core
            is_timm_model = True
        elif "timm/" in path or ".bin" in path or ".pth" in path or ".pt" in path:
            if "jx_vit" not in path and cfg.MODEL.TRANSFORMER_TYPE not in __factory_T_type:
                is_timm_model = True

        if is_timm_model:
            print(f"检测到高级模型权重, 正在使用 timm 构建模型...")

            # --- 1. 自动推断 timm 模型名称 ---
            if "eva02" in path:
                if "large" in path:
                    model_name = "eva02_large_patch14_448.mim_in22k_ft_in1k"
                else:
                    model_name = "eva02_base_patch14_448.mim_in22k_ft_in1k"
            elif "dinov3" in path:
                if "vitl" in path or "large" in path:
                    model_name = "vit_large_patch16_dinov3.lvd1689m"
                else:
                    model_name = "vit_base_patch16_dinov3.lvd1689m"
            elif "PE" in path.lower() or "core" in path.lower():  # PE 模型
                if "large" in path.lower() or "-l-" in path.lower():
                    model_name = "vit_large_patch14_224"
                else:
                    model_name = "vit_base_patch16_224"  # PE-B 通常是 Patch16 (注意: PE模型通常是Patch16)
            else:
                model_name = "vit_base_patch16_224"

            print(f"构建模型: {model_name}")

            # --- 2. 使用 timm 创建 Backbone ---
            is_local_file = os.path.isfile(path)
            model = timm.create_model(model_name, pretrained=not is_local_file, img_size=cfg.INPUT.SIZE_TRAIN, num_classes=0, dynamic_img_size=True)

            # --- 3. 手动加载本地权重 ---
            if is_local_file:
                print(f"正在加载本地权重: {path}")
                checkpoint = torch.load(path, map_location="cpu")

                # 兼容格式
                if "state_dict" in checkpoint:
                    raw_state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    raw_state_dict = checkpoint["model"]
                else:
                    raw_state_dict = checkpoint

                # === 关键修正: 强制检查是否需要映射 ===
                state_dict = raw_state_dict
                if "dinov3" in path:
                    print("检测到 DINOv3，应用 DINOv3 映射...")
                    state_dict = remap_dinov3_checkpoint({"state_dict": raw_state_dict})
                elif "PE" in path.lower() or "core" in path.lower() or "visual.class_embedding" in raw_state_dict:
                    print("检测到 PE/CLIP 结构，应用 PE 映射...")
                    state_dict = remap_pe_checkpoint(raw_state_dict)

                # 自动处理 Pos Embed 尺寸不匹配
                if "pos_embed" in state_dict:
                    ckpt_pos_shape = state_dict["pos_embed"].shape
                    model_pos_shape = model.pos_embed.shape
                    if ckpt_pos_shape != model_pos_shape:
                        print(f"Resize pos_embed: {ckpt_pos_shape} -> {model_pos_shape}")
                        state_dict["pos_embed"] = resample_abs_pos_embed(
                            state_dict["pos_embed"], new_size=model.patch_embed.grid_size, num_prefix_tokens=model.num_prefix_tokens
                        )

                msg = model.load_state_dict(state_dict, strict=False)
                print(f"权重加载结果: {msg}")

                # 再次检查：如果 cls_token 还是丢失，报错提示
                if "cls_token" in msg.missing_keys:
                    print("⚠️ 警告: cls_token 缺失！这意味着权重可能完全没加载进去！请检查映射逻辑。")

            # --- 4. 自动获取维度 ---
            embed_dim = model.num_features
            if hasattr(model, "patch_embed"):
                patch_size = model.patch_embed.patch_size[0]
            else:
                patch_size = 16
            print(f"自动检测特征维度: {embed_dim}, Patch Size: {patch_size}")

            # --- 5. 适配器 Wrapper ---
            class ReIDWrapper(nn.Module):
                def __init__(self, backbone, input_dim, num_classes, cfg, patch_size):
                    super().__init__()
                    self.backbone = backbone
                    self.in_planes = input_dim
                    self.train_pair = False
                    self.logit_scale = nn.Parameter(torch.tensor(2.6592))

                    self.use_cmt = cfg.MODEL.USE_CMT
                    if self.use_cmt:
                        self.h_num = cfg.INPUT.SIZE_TRAIN[0] // patch_size
                        self.w_num = cfg.INPUT.SIZE_TRAIN[1] // patch_size
                        print(f"Wrapper: 启用 CMTModule (Parts={cfg.MODEL.CMT_NUM_PARTS}, Dim={input_dim}, Grid={self.h_num}x{self.w_num})")
                        self.cmt_head = CMTModule(in_dim=self.in_planes, num_parts=cfg.MODEL.CMT_NUM_PARTS, num_classes=num_classes)

                    self.bottleneck = nn.BatchNorm1d(self.in_planes)
                    self.bottleneck.bias.requires_grad_(False)
                    self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)

                    self.bottleneck.apply(weights_init_kaiming)
                    self.classifier.apply(weights_init_classifier)

                def train_with_pair(self):
                    self.train_pair = True

                def train_with_single(self):
                    self.train_pair = False

                def forward(self, x, label=None, cam_label=None, img_wh=None):
                    features = self.backbone.forward_features(x)
                    prefix_tokens = self.backbone.num_prefix_tokens
                    global_feat = features[:, 0]
                    patch_tokens = features[:, prefix_tokens:]

                    if self.use_cmt:
                        f_final, f_comp, cls_scores, attn_map = self.cmt_head(patch_tokens, cam_label, self.h_num, self.w_num)
                        if self.training:
                            part_feats_list = [f_final[:, i, :] for i in range(f_final.size(1))]
                            return cls_scores, part_feats_list, f_comp, attn_map
                        else:
                            return f_final

                    feat = self.bottleneck(global_feat)
                    if self.training:
                        cls_score = self.classifier(feat)
                        return cls_score, global_feat
                    else:
                        return feat

            reid_model = ReIDWrapper(model, input_dim=embed_dim, num_classes=num_class, cfg=cfg, patch_size=patch_size)
            return reid_model

        elif cfg.MODEL.TRANSFORMER_TYPE in __factory_T_type:
            model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
        else:
            raise ValueError(f"Unsupported Transformer type: {cfg.MODEL.TRANSFORMER_TYPE}")
    else:
        model = Backbone(num_class, cfg)
    return model
