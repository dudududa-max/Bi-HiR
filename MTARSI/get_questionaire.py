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

# FGSC-23 Ship Class Label
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

class Args:
    def __init__(self, args):
        self.source = args
        for key, val in args.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.source)
    
def visualize_explanation(
    original_img: Union[np.ndarray, Image.Image],
    expl: np.ndarray,
    algorithm_name: str,
    orig_size: Tuple[int, int],
    true_label: int,
    is_correct: bool,
    correct_count,
    incorrect_count,
    save_path: str = "./questionaire",
    alpha: float = 0.5,
    cmap: str = "jet"
) -> None:
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if isinstance(expl, torch.Tensor):
        expl = expl.squeeze().cpu().numpy()
    
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    true_class = MTARSI_CLASS_NAMES.get(true_label, f"Unknown Class {true_label}")
    
    if orig_size:
        if isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
            orig_size = (int(orig_size[0]), int(orig_size[1]))
            
            original_img = cv2.resize(original_img, orig_size, interpolation=cv2.INTER_CUBIC)
            
            if len(expl.shape) == 4:
                expl = expl[0].transpose(1, 2, 0)
            elif len(expl.shape) == 3:
                expl = expl.transpose(1, 2, 0)
            
            if expl.shape[-1] > 1:  
                expl = np.mean(expl, axis=-1)
            
            expl = cv2.resize(expl, orig_size, interpolation=cv2.INTER_CUBIC)
        else:
            print(f"Warning: orig_size format is incorrect: {orig_size}")
    
    expl = np.maximum(expl, 0)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 0.05], hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(original_img)
    ax1.set_title(f"Class: {true_class}", fontsize=14)
    ax1.axis("off")

    max_val = np.max(expl)
    min_val = 0  
    
    # Overlay the images
    norm_expl = (expl - min_val) / (max_val - min_val + 1e-8)
    
    cmap_obj = plt.get_cmap(cmap)
    colored_expl = cmap_obj(norm_expl)
    
    overlay = np.zeros_like(original_img)
    for c in range(3):  
        overlay[:, :, c] = (1 - alpha) * original_img[:, :, c] + alpha * colored_expl[:, :, c]
    
    ax2.imshow(overlay)
    ax2.set_title(f"{algorithm_name} Explanation", fontsize=14)
    ax2.axis("off")

    os.makedirs(save_path, exist_ok=True)
    
    correctness = "True" if is_correct else "False"
    count = correct_count if is_correct else incorrect_count
    filename = f"{algorithm_name}_{correctness}_{count}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=150)
    plt.close()

def generate_explanations(args, model_type):
    _, test_loader = loader.loaders(1)
    model = loader.load_model().to(device)
    
    os.makedirs("questionaire", exist_ok=True)
    
    for class_id in range(0, 19):  # Select classes 1-19
        class_name = MTARSI_CLASS_NAMES.get(class_id, f"class_{class_id}")
        class_dir = os.path.join("questionaire", class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    for exp_vis in args.ques:
        
        class_counters = {
            class_id: {"correct": 0, "incorrect": 0}
            for class_id in range(0, 20)  
        }
        
        all_classes_complete = False
        
        while not all_classes_complete:
            all_classes_complete = True  
            
            for i, (X, y, orig_sizes, _) in enumerate(test_loader):
                y_true = y.item()
                
                if all(
                    count["correct"] >= 10 and count["incorrect"] >= 10
                    for count in class_counters.values()
                ):
                    all_classes_complete = True
                    break
                
                class_counter = class_counters[y_true]
                
                X, y = X.to(device), y.to(device).long().squeeze()
                
                orig_size = orig_sizes
                if isinstance(orig_size, torch.Tensor):
                    orig_size = orig_size.tolist()
                
                with torch.no_grad():
                    outputs = model(X)
                    pred = outputs.argmax(dim=1)
                
                y_pred = pred.item()
                is_correct = (y_pred == y_true)
                
                if is_correct and class_counter["correct"] >= 10:
                    continue
                if not is_correct and class_counter["incorrect"] >= 10:
                    continue

                expl, _ = get_explanation_pdt(model_type, X, model, pred, exp_vis, sg_r=0.3, sg_N=50)
                
                if is_correct:
                    count = class_counter["correct"]
                    class_counter["correct"] += 1
                else:
                    count = class_counter["incorrect"]
                    class_counter["incorrect"] += 1
                
                class_name = MTARSI_CLASS_NAMES.get(y_true, f"class_{y_true}")
                class_dir = os.path.join("questionaire", class_name)
                
                visualize_explanation(
                    original_img=X,
                    expl=expl,
                    algorithm_name=exp_vis,
                    orig_size=orig_size,
                    true_label=y_true,
                    is_correct=is_correct,
                    correct_count=count,
                    incorrect_count=count,
                    save_path=class_dir, 
                    alpha=0.7
                )
                
                all_classes_complete = all(
                    count["correct"] >= 10 and count["incorrect"] >= 10
                    for count in class_counters.values()
                )
                
                if all_classes_complete:
                    break

# 示例调用（保持不变）
if __name__ == "__main__":
    device = torch.device("cuda")
    
    args = Args(config.args)
    model_type = 'resnet50'
    
    generate_explanations(args, model_type)