# 加载本地许训练权重的CNN网络，然后将权重文件保存到本地

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# 配置设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集路径
# data_dir = '/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset'  # MTARSI数据集
data_dir = '/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset'  # FGSC-23数据集
train_dir = os.path.join(data_dir, 'train')  # 假设你的训练数据在名为 'train' 的子文件夹中
val_dir = os.path.join(data_dir, 'test')      # 假设你的验证数据在名为 'val' 的子文件夹中 (可选，但推荐)

# 检查数据集路径是否存在
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training data directory not found: {train_dir}")
if val_dir and not os.path.exists(val_dir):
    print(f"Warning: Validation data directory not found: {val_dir}")
    val_dir = None

# 数据预处理
image_size = 224
# mean = [0.5497, 0.5578, 0.5460]  # MTARSI数据集
# std = [0.1395, 0.1384, 0.1347]  # MTARSI数据集

mean = [0.3308, 0.3442, 0.3328]  # FGSC-23数据集
std = [0.1913, 0.1849, 0.1884]  # FGSC-23数据集

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

if val_dir:
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
else:
    val_loader = None

# 获取类别数量
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# 加载预训练的 ResNet-18 模型
# pretrained_weights_path = '/STAT/zsk/pretrained_model/pre_checkpoint/hub/checkpoints/resnet18-f37072fd.pth'  # 将此路径替换为你的本地 ResNet-18 预训练权重文件路径
# pretrained_weights_path = '/STAT/zsk/pretrained_model/pre_checkpoint/hub/checkpoints/resnet18-f37072fd.pth'  # 将此路径替换为你的本地 VGG-16 预训练权重文件路径
pretrained_weights_path = '/STAT/zsk/1_pretrained_models/pretrained_weights/resnet50-0676ba61.pth'  # 将此路径替换为你的本地 ResNet-50 预训练权重文件路径
# pretrained_weights_path = '/STAT/zsk/pretrained_model/pre_checkpoint/hub/checkpoints/resnet18-f37072fd.pth'  # 将此路径替换为你的本地 DenseNet-121 预训练权重文件路径
# pretrained_weights_path = '/STAT/zsk/pretrained_model/pre_checkpoint/hub/checkpoints/resnet18-f37072fd.pth'  # 将此路径替换为你的本地 Inception-v3 预训练权重文件路径

# 检查预训练权重文件是否存在
if not os.path.exists(pretrained_weights_path):
    raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_weights_path}")

model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load(pretrained_weights_path))

# 修改最后的全连接层以适应你的数据集的类别数量
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 将模型移动到设备
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 采用 AdamW 优化器，并设置适当的权重衰减以增强模型的泛化能力
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-5)

# 设置余弦退火学习率调度器，T_max 设为总训练周期数，eta_min 为最低学习率
num_epochs = 200
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples * 100
        progress_bar.set_postfix({'loss': f'{epoch_loss:.4f}', 'accuracy': f'{epoch_accuracy:.2f}%'})

    print(f"Epoch {epoch + 1} finished, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

    # 验证 (可选)
    if val_loader:
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct_predictions += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)

        val_epoch_loss = val_loss / val_total_samples
        val_accuracy = val_correct_predictions / val_total_samples * 100
        print(f"Epoch {epoch + 1} Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# 保存训练好的模型权重
save_path = '/STAT/zsk/Remote_sensing_dataset/weights_files/trained_resnet18_FGSC-23_200epochs.pth'  # 设置保存权重的路径
torch.save(model.state_dict(), save_path)
print(f"Trained model weights saved to: {save_path}")