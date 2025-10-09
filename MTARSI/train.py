import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm.data import create_transform, Mixup


# 数据增强和预处理
train_transform = create_transform(
    input_size=224, is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1', interpolation='bicubic')

test_transform = create_transform(input_size=224, is_training=False)

# train_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_test_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# 加载数据集
train_dataset = datasets.ImageFolder(
    root="/DATA/NAS/zjz/FGSC-23/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root="/DATA/NAS/zjz/FGSC-23/test",
    transform=test_transform
)

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

import torch.nn as nn
# from torchvision.models import resnet50, resnet18, vgg16
from networks import convnext_base
model = convnext_base(23)


# from torchvision.models import convnext_base 
# model = convnext_base(weights=None)
# in_dim = model.classifier[-1].in_features
# model.classifier = nn.Sequential(
#     nn.Flatten(1),
#     nn.LayerNorm(in_dim, eps=1e-6),
#     nn.Dropout(0.5),
#     nn.Linear(in_dim, 23)
# )

path = '/DATA/zjz/xai4remote_sensing/models/ConvNeXt_fgsc23_best_ema.pth'
checkpoint = torch.load(path)

model.load_state_dict(checkpoint, strict=False)

# 将模型移到 GPU（如果可用）
device = torch.device("cuda:1")
model = model.to(device)

import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 学习率衰减

# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # 训练阶段
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(True):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 _, preds = torch.max(outputs, 1)

#                 loss.backward()
#                 optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_dataset)
#         epoch_acc = running_corrects.double() / len(train_dataset)
#         print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0

#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             with torch.no_grad():
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 _, preds = torch.max(outputs, 1)

#             val_loss += loss.item() * inputs.size(0)
#             val_corrects += torch.sum(preds == labels.data)

#         val_loss = val_loss / len(val_dataset)
#         val_acc = val_corrects.double() / len(val_dataset)
#         print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

#         # 保存最佳模型
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), './models/mobilevit.pth')

#         scheduler.step()

#     return model

# model = train_model(model, criterion, optimizer, scheduler, num_epochs=50)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

evaluate(model, test_loader)