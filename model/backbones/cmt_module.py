import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. 彻底删除 DiverseEmbeddingGenerator 类 ===
# (此处代码已删除)


# === 2. 保留 ST-CMT 模块 ===
class CMTModule(nn.Module):
    def __init__(self, in_dim, num_parts=2, num_classes=361):
        super(CMTModule, self).__init__()
        self.num_parts = num_parts
        self.in_dim = in_dim

        # MAM: 模态原型
        self.prototypes_rgb = nn.Parameter(torch.randn(num_parts, in_dim))
        self.prototypes_sar = nn.Parameter(torch.randn(num_parts, in_dim))

        self.m_decoder_layer = nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.m_decoder = nn.TransformerDecoder(self.m_decoder_layer, num_layers=1)

        # Topo: 拓扑 Part Tokens
        self.part_tokens = nn.Parameter(torch.randn(1, num_parts, in_dim))
        nn.init.normal_(self.part_tokens, std=0.02)

        # Saliency: 显著性预测
        self.saliency_proj = nn.Sequential(nn.Linear(in_dim, in_dim // 4), nn.ReLU(), nn.Linear(in_dim // 4, 1), nn.Sigmoid())

        # 分类器
        self.classifiers = nn.ModuleList([nn.Linear(in_dim, num_classes, bias=False) for _ in range(num_parts)])

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.prototypes_rgb, std=0.02)
        nn.init.normal_(self.prototypes_sar, std=0.02)

    def get_part_features_soft(self, patch_tokens):
        B, N, D = patch_tokens.shape
        part_queries = self.part_tokens.expand(B, -1, -1)
        attn_score = torch.matmul(part_queries, patch_tokens.transpose(1, 2)) / (D**0.5)
        attn_map = F.softmax(attn_score, dim=-1)
        part_feats = torch.matmul(attn_map, patch_tokens)
        return part_feats, attn_map

    def forward(self, patch_tokens, cam_ids):
        # 1. 软注意力获取特征
        F_parts, attn_map = self.get_part_features_soft(patch_tokens)

        # 2. 计算显著性
        saliency_scores = self.saliency_proj(F_parts)

        # 3. 模态补偿
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

        if self.training:
            cls_outputs = []
            for i in range(self.num_parts):
                cls_outputs.append(self.classifiers[i](F_final[:, i, :]))

            # === 修改处：只返回 5 个值 (移除了 F_parts) ===
            return F_final, F_compensated, cls_outputs, saliency_scores, attn_map

        return F_final.view(F_final.size(0), -1), None, None, None, None
