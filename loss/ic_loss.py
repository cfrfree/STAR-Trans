import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, logits_opt, logits_sar):
        """
        logits_opt: 光学分支分类器的输出 (Before Softmax)
        logits_sar: SAR分支分类器的输出 (Before Softmax)
        """
        # 使用 KL 散度约束两个模态的预测分布一致
        # 温度系数 T=1
        log_prob_opt = F.log_softmax(logits_opt, dim=1)
        prob_sar = F.softmax(logits_sar, dim=1)

        # KL(SAR || Optical)
        loss = F.kl_div(log_prob_opt, prob_sar, reduction="batchmean")
        return loss
