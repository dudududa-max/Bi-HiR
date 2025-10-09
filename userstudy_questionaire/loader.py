import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
# from networks import resnet18, resnet50, VGG16
from timm.data import create_transform, Mixup
# from torchvision.models import convnext_base, resnet50
from networks import resnet50 
from PIL import Image

# 数据增强和预处理
image_size = 224
mean = [0.5497, 0.5578, 0.5460]  # MTARSI数据集
std = [0.1395, 0.1384, 0.1347]  # MTARSI数据集

# mean = [0.3308, 0.3442, 0.3328]  # FGSC-23数据集
# std = [0.1913, 0.1849, 0.1884]  # FGSC-23数据集

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def loaders(batch_size):
    train_dataset = EnhancedImageFolder(
    root="/DATA/NAS/zjz/MTARSI_Dataset/train",
    transform=train_transform)
    test_dataset = EnhancedImageFolder(
    # root="/DATA/NAS/zjz/MTARSI_Dataset/test",
    root="/DATA/NAS/zjz/MTARSI_Dataset/explain",
    transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def load_model(path='/DATA/zjz/MTARSI/models/resnet50_MTARSI.pth'):
    # model = resnet50(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 23)
    model = resnet50(num_classes=20)
    model.load_state_dict(torch.load(path))    
    return model

class EnhancedImageFolder(ImageFolder):
    """增强版 ImageFolder，返回原始图像尺寸和调整后图像尺寸"""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # 预处理：存储所有图像的原始尺寸
        self.orig_sizes = []
        for img_path, _ in self.samples:
            with Image.open(img_path) as img:
                # 存储为 (width, height) 元组
                self.orig_sizes.append(img.size)
    
    def __getitem__(self, index):
        # 获取原始图像路径和标签
        img_path, label = self.samples[index]
        orig_size = self.orig_sizes[index]  # (width, height) 元组
        
        # 加载原始图像并确保为RGB模式
        img = Image.open(img_path)
        
        # 确保图像是三通道
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 保存调整前的尺寸
        pre_transform_size = img.size  # (width, height) 元组
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 获取调整后的尺寸
        if isinstance(img, torch.Tensor):
            # 张量格式: (C, H, W)
            scaled_size = (img.shape[2], img.shape[1])  # (width, height)
        else:
            scaled_size = img.size  # (width, height)
        
        return img, label, orig_size, scaled_size