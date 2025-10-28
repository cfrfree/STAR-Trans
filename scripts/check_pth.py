import torch
import timm

path = "/home/chenfree2002/Python/checkpoints/swin_base_patch4_window7_224_22k.pth"
data = torch.load(path, map_location="cpu")
print("权重顶层键：", list(data.keys()))  # 正常Swin权重顶层键应为[]（直接存储参数）或["state_dict"]


model_pretrain_list = timm.list_models(pretrained=True)
print(len(model_pretrain_list), model_pretrain_list)
