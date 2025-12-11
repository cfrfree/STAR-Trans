import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalConsistencyLoss(nn.Module):
    def __init__(self, h_num=16, w_num=8, margin=2.0):
        super(TopologicalConsistencyLoss, self).__init__()
        self.h_num = h_num
        self.w_num = w_num
        self.margin = margin  # 允许的重叠余量或间隔

    def forward(self, attn_map):
        """
        attn_map: [B, P, N] (N = h_num * w_num)
        P=2, 我们希望 Part 0 在上方 (Y坐标小), Part 1 在下方 (Y坐标大)
        """
        B, P, N = attn_map.shape
        device = attn_map.device

        # 1. 生成每个 Patch 的 Y 坐标网格
        # 形状: [0, 0, ..., 0, 1, 1, ..., 1, ..., 15, 15]
        y_coords = torch.arange(
            self.h_num, dtype=torch.float32, device=device
        ).unsqueeze(1)
        y_coords = y_coords.repeat(1, self.w_num).view(-1)  # [N] (Flatten)

        # 2. 计算每个 Part 的关注中心 (Center of Gravity)
        # attn_map [B, P, N] * y_coords [N] -> [B, P]
        # 这是一个加权平均的过程
        center_y = torch.matmul(attn_map, y_coords)  # [B, P]

        # 3. 计算拓扑顺序 Loss
        # 我们期望: center_y_top (Part 0) < center_y_bottom (Part 1)
        # 即: center_y_top - center_y_bottom < 0
        # 惩罚: ReLU(center_y_top - center_y_bottom + margin)

        center_y_top = center_y[:, 0]
        center_y_bottom = center_y[:, 1]

        # 如果 top 在 bottom 下面 (差值>0)，就产生 Loss
        loss_topo = F.relu(center_y_top - center_y_bottom + self.margin).mean()

        return loss_topo
