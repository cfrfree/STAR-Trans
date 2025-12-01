import torch
import torch.nn as nn
import torch.nn.functional as F


class CPMLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CPMLoss, self).__init__()
        self.margin = margin

    def forward(self, f_original, f_generated, pids, camids):
        """
        计算 Center-Guided Pair Mining Loss (CPM Loss)

        Args:
            f_original (Tensor): 原始特征 (ST-CMT输出), 形状 [B, P, D]
                                 如果是 List，函数内部会自动堆叠。
            f_generated (Tensor): 生成的多样性特征 (UFE输出), 形状 [B, K, P, D]
                                  K 是生成的变体数量 (num_variations)
            pids (Tensor): ID 标签, 形状 [B]
            camids (Tensor): 模态标签, 形状 [B] (0: RGB, 1: SAR)

        Returns:
            loss (Tensor): 标量 Loss
        """
        # 1. 输入数据清洗
        if f_generated is None:
            return torch.tensor(0.0).to(pids.device)

        # 兼容性处理：如果 f_original 是列表 (来自 make_model 的 part_feats_list)
        if isinstance(f_original, list):
            f_original = torch.stack(f_original, dim=1)  # [B, P, D]

        # 确保数据都在同一设备
        device = f_original.device
        if f_generated.device != device:
            f_generated = f_generated.to(device)

        B, K, P, D = f_generated.shape
        loss = 0.0
        valid_part_count = 0  # 记录有效计算的 Part 数量

        # 2. 遍历每个 Part 分别计算
        # 因为我们是对齐具有物理意义的 Part (如船头对船头)
        for p in range(P):
            f_orig_p = f_original[:, p, :]  # [B, D]
            f_gen_p = f_generated[:, :, p, :]  # [B, K, D]

            part_loss = 0.0
            id_count = 0

            unique_pids = torch.unique(pids)
            for pid in unique_pids:
                # 找到当前 ID 的样本索引
                indices = pids == pid
                idx_rgb = indices & (camids == 0)
                idx_sar = indices & (camids == 1)

                # 如果该 ID 缺少任一模态的数据，无法计算跨模态中心，跳过
                if not (idx_rgb.any() and idx_sar.any()):
                    continue

                id_count += 1

                # === Step A: 计算真实模态中心 (Centroids) ===
                # center_rgb: 当前 ID 所有 RGB 样本的中心 [D]
                center_rgb = f_orig_p[idx_rgb].mean(dim=0).detach()
                # center_sar: 当前 ID 所有 SAR 样本的中心 [D]
                center_sar = f_orig_p[idx_sar].mean(dim=0).detach()

                # === Step B: 核心 Loss 计算 (Pull Cross, Push Self) ===

                # --- 情况 1: RGB 样本生成的变体 ---
                if idx_rgb.any():
                    # 取出当前 ID 下所有 RGB 样本生成的 K 个变体 [N_rgb, K, D]
                    gen_from_rgb = f_gen_p[idx_rgb]
                    # 取出对应的原始 RGB 特征 [N_rgb, 1, D] (unsqueeze 用于广播)
                    orig_rgb = f_orig_p[idx_rgb].unsqueeze(1)

                    # 1. Pull: 生成特征应靠近 SAR 中心 (跨模态对齐)
                    # center_sar 扩展为 [1, 1, D]
                    d_pull = (gen_from_rgb - center_sar.view(1, 1, -1)).pow(2).sum(dim=-1).sqrt()

                    # 2. Push: 生成特征应远离 原始 RGB 特征 (多样性扩张)
                    d_push = (gen_from_rgb - orig_rgb).pow(2).sum(dim=-1).sqrt()

                    # Loss = max(0, d_pull - d_push + margin)
                    # 含义: d_pull (跨模态距离) 应该比 d_push (生成距离) 小至少 margin
                    part_loss += F.relu(d_pull - d_push + self.margin).mean()

                # --- 情况 2: SAR 样本生成的变体 ---
                if idx_sar.any():
                    gen_from_sar = f_gen_p[idx_sar]
                    orig_sar = f_orig_p[idx_sar].unsqueeze(1)

                    # Pull: 靠近 RGB 中心
                    d_pull = (gen_from_sar - center_rgb.view(1, 1, -1)).pow(2).sum(dim=-1).sqrt()

                    # Push: 远离 原始 SAR 特征
                    d_push = (gen_from_sar - orig_sar).pow(2).sum(dim=-1).sqrt()

                    part_loss += F.relu(d_pull - d_push + self.margin).mean()

            # 只有当该 Part 有有效的 ID 对时才计入 Loss
            if id_count > 0:
                loss += part_loss / id_count
                valid_part_count += 1

        # 平均化所有 Part 的 Loss
        if valid_part_count > 0:
            loss /= valid_part_count

        return loss
