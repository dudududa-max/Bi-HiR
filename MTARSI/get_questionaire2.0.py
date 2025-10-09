import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from PIL import Image
from typing import Union, Tuple
from metrics_utils import get_explanation_pdt
import loader
import config
import eval_metrics as eis 

# FGSC-23舰船类别标签
MTARSI_CLASS_NAMES = {
    0: "A-10",
    1: "A-26",
    2: "B-1",
    3: "B-2",
    4: "B-29",
    5: "B-52",
    6: "Boeing",
    7: "C-5",
    8: "C-17",
    9: "C-21",
    10: "C-130",
    11: "C-135",
    12: "E-3",
    13: "F-16",
    14: "F-22",
    15: "KC-10",
    16: "P-63",
    17: "T-6",
    18: "T-43",
    19: "U-2"
}

def visualize_explanation(
    original_img: Union[np.ndarray, Image.Image],
    expl: np.ndarray,
    algorithm_name: str,
    orig_size: Tuple[int, int],
    true_label: int,
    is_correct: bool,
    correct_count,
    incorrect_count,
    save_path: str = "./questionaire_explain",
    alpha: float = 0.5,
    cmap: str = "jet"
) -> None:
    # 处理4D张量输入
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if isinstance(expl, torch.Tensor):
        expl = expl.squeeze().cpu().numpy()
    
    # 归一化处理
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    true_class = MTARSI_CLASS_NAMES.get(true_label, f"Unknown Class {true_label}")
    
    # 如果提供了原始尺寸，将图像和热力图还原到原始尺寸
    if orig_size:
        # 确保 orig_size 是 (width, height) 格式
        if isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
            # 转换为整数元组
            orig_size = (int(orig_size[0]), int(orig_size[1]))
            
            # 还原原始图像
            original_img_resized = cv2.resize(original_img, orig_size, interpolation=cv2.INTER_CUBIC)
            
            # 处理解释图
            if len(expl.shape) == 4:
                expl = expl[0].transpose(1, 2, 0)
            elif len(expl.shape) == 3:
                expl = expl.transpose(1, 2, 0)
            
            if expl.shape[-1] > 1:  # 如果是多通道，取平均值
                expl = np.mean(expl, axis=-1)
            
            # 将热力图还原到原始尺寸
            expl_resized = cv2.resize(expl, orig_size, interpolation=cv2.INTER_CUBIC)
        else:
            print(f"警告: orig_size 格式不正确: {orig_size}")
    else:
        # 如果没有提供原始尺寸，使用原始尺寸
        original_img_resized = original_img
        expl_resized = expl
    
    # -------------------------- 创建原始图像和解释图的对比图像 --------------------------
    # 1. 原始图像
    fig_orig = plt.figure(figsize=(6, 6))
    plt.imshow(original_img_resized)
    # plt.title(f"原始图像: {true_class}")
    plt.axis("off")
    
    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    
    # 生成文件名
    correctness_eng = "True" if is_correct else "False"
    count = correct_count if is_correct else incorrect_count
    orig_filename = f"original_img_{correctness_eng}_{count}.png"
    plt.savefig(os.path.join(save_path, orig_filename), bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig_orig)
    
    # 2. 解释热力图
    # 计算颜色范围
    max_val = np.max(expl_resized)
    min_val = 0  # 最小值设为0，忽略负数
    
    # 创建叠加图像
    norm_expl = (expl_resized - min_val) / (max_val - min_val + 1e-8)
    cmap_obj = plt.get_cmap(cmap)
    colored_expl = cmap_obj(norm_expl)
    
    # 创建叠加图像 (alpha混合)
    overlay = np.zeros_like(original_img_resized)
    for c in range(3):  # RGB通道
        overlay[:, :, c] = (1 - alpha) * original_img_resized[:, :, c] + alpha * colored_expl[:, :, c]
    
    # 绘制解释热力图
    fig_expl = plt.figure(figsize=(8, 8))
    gs = fig_expl.add_gridspec(2, 1, height_ratios=[0.9, 0.1], hspace=0.05)
    ax1 = fig_expl.add_subplot(gs[0, 0])
    # cbar_ax = fig_expl.add_subplot(gs[1, :])

    
    ax1.imshow(overlay)
    
    # 设置标题
    correctness = "True" if is_correct else "False"
    # title = f"{algorithm_name}解释: {true_class} ({correctness})"
    # ax1.set_title(title, fontsize=14)
    ax1.axis("off")
    
    # # 添加颜色条 - 只显示正数范围 (0到max_val)
    # ColorbarBase(
    #     cbar_ax,
    #     cmap=cmap_obj,
    #     norm=Normalize(vmin=0, vmax=max_val),  # 只显示0到最大值
    #     orientation="horizontal",
    # )
    # cbar_ax.set_xlabel("Feature Importance", fontsize=10)  # 更新标签
    # cbar_ax.xaxis.set_ticks([0, max_val])
    # cbar_ax.xaxis.set_ticklabels(["Low", "High"], fontsize=9)
    
    # 生成解释图文件名
    expl_filename = f"{algorithm_name}_{correctness_eng}_{count}.png"
    plt.savefig(os.path.join(save_path, expl_filename), bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig_expl)

# 修改后的主流程（保持不变）
def generate_explanations(args, model_type):
    _, test_loader = loader.loaders(1)
    model = loader.load_model().to(device)
    model.eval()
    
    # 创建问卷根目录
    os.makedirs("questionaire_explain", exist_ok=True)
    
    # 为每个类别创建独立文件夹
    for class_id in range(0, 20):  # 类别1到22
        class_name = MTARSI_CLASS_NAMES.get(class_id, f"class_{class_id}")
        class_dir = os.path.join("questionaire_explain", class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    for exp_vis in args.ques:
        print(f"\n为解释方法 '{exp_vis}' 生成可视化...")
        
        # 初始化每个类别的计数器
        class_counters = {
            class_id: {"correct": 0, "incorrect": 0}
            for class_id in range(0, 20)  
        }
        

        for i, (X, y, orig_sizes, _) in enumerate(test_loader):
            y_true = y.item()
            
            # 检查是否所有类别都已完成收集
            if all(
                count["correct"] >= 5 and count["incorrect"] >= 10
                for count in class_counters.values()
            ):
                print("所有类别都已完成样本收集")
                break
            
            # 获取当前类别的计数器
            class_counter = class_counters[y_true]
            
            X, y = X.to(device), y.to(device).long().squeeze()
            
            # 确保 orig_sizes 是 (width, height) 格式
            orig_size = orig_sizes
            if isinstance(orig_size, torch.Tensor):
                orig_size = orig_size.tolist()
            
            # 获取模型预测
            with torch.no_grad():
                outputs = model(X)
                pred = outputs.argmax(dim=1)
            
            y_pred = pred.item()
            # print(y_pred)
            # print(y_true)
            is_correct = (y_pred == y_true)
            # 检查当前类别收集状态
            if is_correct and class_counter["correct"] >= 5:
                continue
            if not is_correct and class_counter["incorrect"] >= 10:
                continue
            
            # 获取解释图
            expl, _ = get_explanation_pdt(model_type, X, model, pred, exp_vis, sg_r=0.3, sg_N=50)
            
            # 确定计数（正确或错误）
            if is_correct:
                count = class_counter["correct"]
                class_counter["correct"] += 1
            else:
                count = class_counter["incorrect"]
                class_counter["incorrect"] += 1
            
            # 获取当前类别的保存目录
            class_name = MTARSI_CLASS_NAMES.get(y_true, f"class_{y_true}")
            class_dir = os.path.join("questionaire_explain", class_name)
            
            # 可视化处理 - 指定类别的保存目录
            visualize_explanation(
                original_img=X,
                expl=expl,
                algorithm_name=exp_vis,
                orig_size=orig_size,
                true_label=y_true,
                is_correct=is_correct,
                correct_count=count,
                incorrect_count=count,
                save_path=class_dir,  # 保存到类别目录
                alpha=0.7
            )

# 示例调用（保持不变）
if __name__ == "__main__":
    device = torch.device("cuda:3")
    print(f"使用设备: {device}")
    
    args = eis.Args(config.args)
    model_type = 'resnet50'
    
    generate_explanations(args, model_type)