import torch

# 你的权重文件路径
path = "logs/pretrain_base/transformer_40.pth"

# 加载权重
checkpoint = torch.load(path, map_location="cpu")

# 1. 自动判断权重的结构（有些pth文件会多包一层 'model' 或 'state_dict'）
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("检测到 'state_dict' 键，已提取子字典。")
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("检测到 'model' 键，已提取子字典。")
else:
    state_dict = checkpoint
    print("未检测到嵌套结构，直接作为 state_dict 处理。")

# 2. 打印所有的 Key
print(f"\n=== 总共有 {len(state_dict)} 个 Key ===")
for key in state_dict.keys():
    print(key)
