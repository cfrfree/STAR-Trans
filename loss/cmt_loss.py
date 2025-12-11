# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ModalityConsistencyLoss(nn.Module):
#     def __init__(self):
#         super(ModalityConsistencyLoss, self).__init__()

#     def forward(self, f_final, f_comp, pids, camids):
#         """
#         f_final: [B, P, D] Tensor 或者 List of [B, D] Tensors
#         f_comp:  [B, P, D] Tensor
#         pids:    [B]
#         camids:  [B]
#         """
#         # === 修复开始：如果 f_final 是列表，将其堆叠回 Tensor ===
#         if isinstance(f_final, list):
#             # 将包含 P 个 [B, D] Tensor 的列表堆叠为 [B, P, D]
#             f_final = torch.stack(f_final, dim=1)
#         # === 修复结束 ===

#         loss_cyc = 0.0
#         # 确保 pids 和 camids 都在同一设备上
#         pids = pids.to(f_final.device)
#         camids = camids.to(f_final.device)

#         unique_pids = torch.unique(pids)

#         count = 0
#         for pid in unique_pids:
#             # 找到当前 Batch 中属于该 ID 的所有样本
#             indices = pids == pid

#             # 分离出 RGB 和 SAR 的样本索引
#             idx_rgb = indices & (camids == 0)
#             idx_sar = indices & (camids == 1)

#             # 只有当一个 ID 同时拥有 RGB 和 SAR 样本时，才能计算一致性损失
#             if idx_rgb.any() and idx_sar.any():
#                 # 1. 计算真实的模态中心 (Centroids)
#                 # f_final 已经是 [B, P, D] Tensor 了，可以使用布尔索引
#                 real_center_rgb = f_final[idx_rgb].mean(dim=0).detach()
#                 real_center_sar = f_final[idx_sar].mean(dim=0).detach()

#                 # 2. 获取补偿特征
#                 generated_sar = f_comp[idx_rgb]  # RGB 样本生成的假 SAR
#                 generated_rgb = f_comp[idx_sar]  # SAR 样本生成的假 RGB

#                 # 3. 计算 MSE Loss
#                 loss_cyc += F.mse_loss(generated_sar, real_center_sar.unsqueeze(0).expand_as(generated_sar))
#                 loss_cyc += F.mse_loss(generated_rgb, real_center_rgb.unsqueeze(0).expand_as(generated_rgb))
#                 count += 1

#         if count > 0:
#             loss_cyc /= count

#         return loss_cyc
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityConsistencyLoss(nn.Module):
    def __init__(self):
        super(ModalityConsistencyLoss, self).__init__()

    def forward(self, f_final, f_comp, pids, camids, saliency_scores=None):
        """
        f_final: [B, P, D] Tensor
        f_comp:  [B, P, D] Tensor
        pids:    [B]
        camids:  [B]
        saliency_scores: [B, P, 1] Tensor (新增参数, 范围0-1)
        """
        # === 修复: 如果 f_final 是列表，将其堆叠回 Tensor ===
        if isinstance(f_final, list):
            f_final = torch.stack(f_final, dim=1)

        loss_cyc = 0.0
        # 确保数据在同一设备
        pids = pids.to(f_final.device)
        camids = camids.to(f_final.device)
        if saliency_scores is not None:
            saliency_scores = saliency_scores.to(f_final.device)

        unique_pids = torch.unique(pids)

        count = 0
        for pid in unique_pids:
            # 找到当前 Batch 中属于该 ID 的所有样本
            indices = pids == pid

            # 分离出 RGB 和 SAR 的样本索引
            idx_rgb = indices & (camids == 0)
            idx_sar = indices & (camids == 1)

            # 只有当一个 ID 同时拥有 RGB 和 SAR 样本时，才能计算一致性损失
            if idx_rgb.any() and idx_sar.any():
                # 1. 计算真实的模态中心 (Centroids)
                real_center_rgb = f_final[idx_rgb].mean(dim=0).detach()  # [P, D]
                real_center_sar = f_final[idx_sar].mean(dim=0).detach()  # [P, D]

                # 2. 获取补偿特征 & 显著性分数
                generated_sar = f_comp[idx_rgb]  # RGB 样本生成的假 SAR [N_rgb, P, D]
                generated_rgb = f_comp[idx_sar]  # SAR 样本生成的假 RGB [N_sar, P, D]

                # 3. 计算加权 MSE Loss
                # 注意：不能直接用 F.mse_loss，因为它会直接求平均，无法加权

                # --- Part A: 假 SAR vs 真 SAR 中心 ---
                # 计算逐元素的平方误差: [N_rgb, P, D]
                diff_sar = generated_sar - real_center_sar.unsqueeze(0).expand_as(
                    generated_sar
                )
                sq_err_sar = diff_sar.pow(2)

                # 如果有显著性分数，则加权
                if saliency_scores is not None:
                    # 获取 RGB 样本对应的显著性 [N_rgb, P, 1]
                    # 逻辑：如果这个 RGB 样本的某个 Part 很重要，生成的 SAR 特征也必须对齐得很准
                    w_rgb = saliency_scores[idx_rgb]
                    sq_err_sar = sq_err_sar * w_rgb

                loss_cyc += sq_err_sar.mean()  # 对 N, P, D 求平均

                # --- Part B: 假 RGB vs 真 RGB 中心 ---
                diff_rgb = generated_rgb - real_center_rgb.unsqueeze(0).expand_as(
                    generated_rgb
                )
                sq_err_rgb = diff_rgb.pow(2)

                if saliency_scores is not None:
                    # 获取 SAR 样本对应的显著性 [N_sar, P, 1]
                    w_sar = saliency_scores[idx_sar]
                    sq_err_rgb = sq_err_rgb * w_sar

                loss_cyc += sq_err_rgb.mean()

                count += 1

        if count > 0:
            loss_cyc /= count

        # === 防止坍塌的正则项 (可选但推荐) ===
        # 如果显著性分数完全由 Loss 决定，网络可能会倾向于输出全 0 来让 Loss 变小。
        # 增加一个正则项，鼓励显著性分数尽可能大（接近 1），除非真的是噪声。
        if saliency_scores is not None:
            # Regularization: -mean(log(saliency)) or mean(1 - saliency)
            # 这里的权重 0.1 可以调节，防止网络偷懒把所有权重都变成0
            reg_loss = (1.0 - saliency_scores).mean()
            return loss_cyc + 0.1 * reg_loss

        return loss_cyc
