import torch
from collections import OrderedDict

# 加载pth文件（添加weights_only=True规避警告，模型权重文件通常安全）
checkpoint = torch.load(
    "/home/share/chenfree/ReID/swin_base_patch4_window7_224_22k.pth", map_location="cpu", weights_only=True  # 按警告提示添加，避免未来版本兼容问题
)

# 解析顶层字典
print("顶层字典结构：")
for top_key, top_value in checkpoint.items():
    print(f"\n顶层键：{top_key}")
    # 若值是OrderedDict（通常存储模型层权重），则进一步解析
    if isinstance(top_value, OrderedDict):
        print("  包含的层权重信息：")
        for layer_key, weight in top_value.items():
            if isinstance(weight, torch.Tensor):
                print(f"  - 层键：{layer_key} | 张量尺寸：{weight.shape}")
            else:
                print(f"  - 层键：{layer_key} | 类型：{type(weight)}（非张量）")
    else:
        print(f"  类型：{type(top_value)}")
