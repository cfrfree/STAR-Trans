import torch
import torch.nn as nn


class ChannelFCE(nn.Module):
    def __init__(self, in_dim):
        super(ChannelFCE, self).__init__()
        # 1. 模态交互：计算 "受损程度" 或 "融合门控"
        # 输入: [Optical_Feat, SAR_Prototype] -> 输出: Channel Mask (Gate)
        self.gate_generator = nn.Sequential(nn.Linear(in_dim * 2, in_dim // 4), nn.ReLU(), nn.Linear(in_dim // 4, in_dim), nn.Sigmoid())

    def forward(self, feat_opt, feat_sar_proto):
        """
        feat_opt: [B, D] - 当前 batch 的光学图像特征
        feat_sar_proto: [B, D] - 对应 ID 的 SAR 模态原型 (Centroid)
        """
        # 拼接两者，判断哪些通道 Optical 比较弱，需要 SAR 来补
        cat_feat = torch.cat([feat_opt, feat_sar_proto], dim=1)  # [B, 2D]

        # 生成通道门控掩码 M
        # M 越接近 1，表示该通道主要信赖 SAR；M 越接近 0，表示信赖 Optical 自身
        mask = self.gate_generator(cat_feat)  # [B, D]

        # 执行增强 (Enhancement)
        # 类似于 FACENet: Out = SAR * M + Optical * (1 - M)
        feat_enhanced = feat_sar_proto * mask + feat_opt * (1 - mask)

        return feat_enhanced, mask
