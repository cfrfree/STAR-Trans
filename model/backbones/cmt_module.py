# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy


# class CMTModule(nn.Module):
#     def __init__(self, in_dim, num_parts=6, num_classes=361):
#         super(CMTModule, self).__init__()
#         self.num_parts = num_parts
#         self.in_dim = in_dim

#         # === 1. MAM: 模态级对齐模块 ===
#         self.prototypes_rgb = nn.Parameter(torch.randn(num_parts, in_dim))
#         self.prototypes_sar = nn.Parameter(torch.randn(num_parts, in_dim))

#         self.m_decoder_layer = nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
#         self.m_decoder = nn.TransformerDecoder(self.m_decoder_layer, num_layers=1)

#         # === 2. IAM & Saliency: 实例级对齐 & 显著性预测 ===
#         # 参数生成器 (生成 gamma 和 beta) - 這是原有的 IAM
#         self.param_generator = nn.Sequential(nn.Linear(in_dim, in_dim // 4), nn.ReLU(), nn.Linear(in_dim // 4, in_dim * 2))

#         # === NEW: 散射显著性预测器 (Saliency Projector) ===
#         # 输入 Part 特征，输出该 Part 的重要性权重 (0~1)
#         # 它可以学习识别：这个 Part 是“强散射的舰船结构”还是“无用的海面背景”
#         self.saliency_proj = nn.Sequential(nn.Linear(in_dim, in_dim // 4), nn.ReLU(), nn.Linear(in_dim // 4, 1), nn.Sigmoid())  # 限制在 0-1 之间

#         # === 3. 分类器 ===
#         self.classifiers = nn.ModuleList([nn.Linear(in_dim, num_classes, bias=False) for _ in range(num_parts)])

#         self._init_params()

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.001)
#         nn.init.normal_(self.prototypes_rgb, std=0.02)
#         nn.init.normal_(self.prototypes_sar, std=0.02)

#     def get_part_features(self, patch_tokens, h_num, w_num):
#         """
#         将 ViT 的 patch tokens 转换为 Part 特征
#         """
#         B, N, D = patch_tokens.shape
#         spatial_feat = patch_tokens.view(B, h_num, w_num, D)

#         rows_per_part = h_num // self.num_parts

#         part_feats = []
#         for i in range(self.num_parts):
#             part_rows = spatial_feat[:, i * rows_per_part : (i + 1) * rows_per_part, :, :]
#             part_feat = part_rows.mean(dim=(1, 2))
#             part_feats.append(part_feat)

#         return torch.stack(part_feats, dim=1)  # [B, P, D]

#     def forward(self, patch_tokens, cam_ids, h_num, w_num):
#         # 1. 获取 Part 特征 F
#         F_parts = self.get_part_features(patch_tokens, h_num, w_num)  # [B, P, D]

#         # === NEW: 计算显著性权重 (Saliency Calculation) ===
#         # [B, P, D] -> [B, P, 1]
#         # 这个权重将告诉 Loss 函数：哪些 Part 需要被强对齐，哪些可以忽略
#         saliency_scores = self.saliency_proj(F_parts)

#         # 2. 模态补偿 (MAM)
#         B = F_parts.size(0)
#         is_rgb = cam_ids == 0
#         is_sar = cam_ids == 1

#         F_compensated = torch.zeros_like(F_parts)

#         # --- 对 RGB 样本进行 SAR 特征补偿 ---
#         if is_rgb.any():
#             rgb_feats = F_parts[is_rgb]
#             tgt = self.prototypes_sar.unsqueeze(1).expand(-1, rgb_feats.size(0), -1)
#             memory = rgb_feats.permute(1, 0, 2)
#             out_sar_comp = self.m_decoder(tgt, memory)
#             F_compensated[is_rgb] = out_sar_comp.permute(1, 0, 2)

#         # --- 对 SAR 样本进行 RGB 特征补偿 ---
#         if is_sar.any():
#             sar_feats = F_parts[is_sar]
#             tgt = self.prototypes_rgb.unsqueeze(1).expand(-1, sar_feats.size(0), -1)
#             memory = sar_feats.permute(1, 0, 2)
#             out_rgb_comp = self.m_decoder(tgt, memory)
#             F_compensated[is_sar] = out_rgb_comp.permute(1, 0, 2)

#         # 最终特征
#         F_final = F_parts + F_compensated

#         # === 训练阶段返回 Logits 用于计算 ID Loss ===
#         if self.training:
#             cls_outputs = []
#             for i in range(self.num_parts):
#                 cls_outputs.append(self.classifiers[i](F_final[:, i, :]))

#             # === NEW: 增加返回 saliency_scores ===
#             # 将显著性分数传出去，以便在 loss/cmt_loss.py 中进行加权
#             return F_final, F_compensated, cls_outputs, saliency_scores

#         # === 测试阶段 ===
#         # 测试阶段不需要 saliency，保持对齐即可
#         return F_final.view(B, -1), None, None, None

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMTModule(nn.Module):
    def __init__(self, in_dim, num_parts=2, num_classes=361):  # 建议默认 num_parts=2
        super(CMTModule, self).__init__()
        self.num_parts = num_parts
        self.in_dim = in_dim

        # === 1. MAM: 模态级对齐模块 (保持不变) ===
        self.prototypes_rgb = nn.Parameter(torch.randn(num_parts, in_dim))
        self.prototypes_sar = nn.Parameter(torch.randn(num_parts, in_dim))

        self.m_decoder_layer = nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.m_decoder = nn.TransformerDecoder(self.m_decoder_layer, num_layers=1)

        # === 2. NEW: 双向拓扑一致性 (BTC) ===
        # 定义可学习的 Part Tokens (类似 DETR 的 Object Queries)
        # 形状: [1, P, D]。它们会自动学习去关注"船头"和"船尾"
        self.part_tokens = nn.Parameter(torch.randn(1, num_parts, in_dim))
        nn.init.normal_(self.part_tokens, std=0.02)

        # === 3. 分类器 (保持不变) ===
        self.classifiers = nn.ModuleList([nn.Linear(in_dim, num_classes, bias=False) for _ in range(num_parts)])

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.prototypes_rgb, std=0.02)
        nn.init.normal_(self.prototypes_sar, std=0.02)
        # 注意：part_tokens 已经在上面初始化了

    def get_part_features_soft(self, patch_tokens):
        """
        使用 Attention 机制软聚合特征
        输入: patch_tokens [B, N, D] (N=128)
        输出: part_feats [B, P, D], attn_map [B, P, N]
        """
        B, N, D = patch_tokens.shape

        # 扩展 part_tokens 到 Batch 维度: [1, P, D] -> [B, P, D]
        part_queries = self.part_tokens.expand(B, -1, -1)

        # 计算注意力分数 (Attention Scores)
        # Q: part_queries [B, P, D]
        # K: patch_tokens [B, N, D]
        # Matmul -> [B, P, N]
        attn_score = torch.matmul(part_queries, patch_tokens.transpose(1, 2))
        attn_score = attn_score / (D**0.5)  # 缩放

        # Softmax 归一化得到注意力图 (每个 Part 关注哪些 Patch)
        attn_map = F.softmax(attn_score, dim=-1)  # [B, P, N]

        # 加权聚合特征
        # V: patch_tokens [B, N, D]
        # [B, P, N] x [B, N, D] -> [B, P, D]
        part_feats = torch.matmul(attn_map, patch_tokens)

        return part_feats, attn_map

    def forward(self, patch_tokens, cam_ids, h_num, w_num):
        # 1. 获取软聚合的 Part 特征
        # 注意：不再需要 view 成 h_num/w_num 来硬切了
        F_parts, attn_map = self.get_part_features_soft(patch_tokens)  # [B, P, D], [B, P, N]

        # 2. 模态补偿 (MAM) - 逻辑完全不变
        is_rgb = cam_ids == 0
        is_sar = cam_ids == 1
        F_compensated = torch.zeros_like(F_parts)

        if is_rgb.any():
            rgb_feats = F_parts[is_rgb]
            tgt = self.prototypes_sar.unsqueeze(1).expand(-1, rgb_feats.size(0), -1)
            memory = rgb_feats.permute(1, 0, 2)
            out_sar_comp = self.m_decoder(tgt, memory)
            F_compensated[is_rgb] = out_sar_comp.permute(1, 0, 2).to(F_compensated.dtype)

        if is_sar.any():
            sar_feats = F_parts[is_sar]
            tgt = self.prototypes_rgb.unsqueeze(1).expand(-1, sar_feats.size(0), -1)
            memory = sar_feats.permute(1, 0, 2)
            out_rgb_comp = self.m_decoder(tgt, memory)
            F_compensated[is_sar] = out_rgb_comp.permute(1, 0, 2).to(F_compensated.dtype)

        F_final = F_parts + F_compensated

        # === 3. 返回结果 ===
        if self.training:
            cls_outputs = []
            for i in range(self.num_parts):
                cls_outputs.append(self.classifiers[i](F_final[:, i, :]))

            # === NEW: 返回 attn_map 用于计算拓扑 Loss ===
            return F_final, F_compensated, cls_outputs, attn_map

        # 测试阶段
        return F_final.view(F_final.size(0), -1), None, None, None
