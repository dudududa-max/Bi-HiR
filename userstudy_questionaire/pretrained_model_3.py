#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGSC-23 ResNet-50 – 最小修改组合
• 在线 RandAugment + Mixup/CutMix + SoftTargetCrossEntropy
• 全层微调（不再冻结 40 个参数）
• fc 前加 Dropout
• AdamW  lr=1e-4 + CosineAnnealing + EarlyStopping
• ExponentialMovingAverage(EMA)
"""

import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from torch_ema import ExponentialMovingAverage
from torchvision.models import convnext_base 

# ---------------------------- 0  环境 ----------------------------
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('>>> Using device:', device)

# ---------------------------- 1  数据 ----------------------------
data_dir = '//DATA/NAS/zjz/FGSC-23'
train_dir = os.path.join(data_dir, 'train')   # >>> 变更：用原始 train
val_dir   = os.path.join(data_dir, 'test')

# 1.1 在线增强（RandAugment 已含随机水平翻转、裁剪等）
train_tf = create_transform(
    input_size=224, is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1', interpolation='bicubic')
val_tf = create_transform(input_size=224, is_training=False)

train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)
num_classes = len(train_ds.classes)
print('>>> #Classes =', num_classes)

# 1.2 Mixup / CutMix 生成软标签
mixup_fn = Mixup(
    mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1,
    num_classes=num_classes)

# ---------------------------- 2  模型 ----------------------------
w_path = '/DATA/zjz/xai4remote_sensing/models/ConvNeXt_fgsc23_best_ema.pth'

model = convnext_base(weights=None)

state = torch.load(w_path, map_location='cpu')
# 如果是 checkpoint，请先取出其中的 "model" 或 "state_dict" 字段
if 'state_dict' in state:       # 兼容一般 checkpoint 格式
    state = {k.replace('module.', ''): v          # 清理可能的 DataParallel 前缀
             for k, v in state['state_dict'].items()}
model.load_state_dict(state, strict=False)        # strict=False 忽略分类头

in_dim = model.classifier[-1].in_features
model.classifier = nn.Sequential(
    nn.Flatten(1),
    nn.LayerNorm(in_dim, eps=1e-6),
    nn.Dropout(0.2),
    nn.Linear(in_dim, num_classes)
)

# 冻结前3阶段权重，只训练分类头和最后阶段
# for name, param in model.named_parameters():
#     if "stage4" not in name and "classifier" not in name:
#         param.requires_grad = False

model = model.to(device)

# ---------------------------- 3  损失、优化、EMA ------------------
criterion = SoftTargetCrossEntropy()          # >>> 变更
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-5)  # >>> 更小 LR
sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)  # >>> EMA

# ---------------------------- 4  Early-Stopping ------------------
patience, best_acc, epochs_no_improve = 300, 0.0, 0

# ---------------------------- 5  训练 ----------------------------
for epoch in range(1, 301):
    # ---------- Train ----------
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    prog = tqdm(train_loader, desc=f'Epoch {epoch}/300')
    for imgs, labels in prog:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = mixup_fn(imgs, labels)          # >>> 生成软标签

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ema.update()                                   # >>> EMA 更新

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        # 软标签无法计算准确率，这里仅展示 loss
        total += imgs.size(0)
        prog.set_postfix(loss=f'{running_loss/total:.4f}')

    sched.step()

    # ---------- Val ----------
    model.eval()
    v_correct, v_total = 0, 0
    with ema.average_parameters():                     # >>> 用 EMA 权重评估
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)
    v_acc = 100 * v_correct / v_total
    print(f'>> Val Acc {v_acc:.2f}%')

    # Early-Stopping
    if v_acc > best_acc:
        best_acc, epochs_no_improve = v_acc, 0
        torch.save(model.state_dict(),
                   '/DATA/zjz/xai4remote_sensing/models/ConvNeXt_fgsc23_best_ema_zjz.pth')
        print(f'** Best model saved @ {best_acc:.2f}% **')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping triggered!')
            break

print(f'Finished! Best Val Acc = {best_acc:.2f}%')
