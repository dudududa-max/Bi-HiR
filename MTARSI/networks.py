import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Union, List, Optional
import torch.nn.functional as F


# 定义 Bottleneck 残差块（与官方实现完全一致）
class Bottleneck(nn.Module):
    expansion: int = 4  # 通道扩展倍数

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # 三个卷积层（命名与官方一致）
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion: int = 1  # 通道无扩展

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 两个卷积层（命名与官方一致）
        self.conv1 = nn.Conv2d(
            inplanes, 
            planes, 
            kernel_size=3, 
            stride=stride, 
            padding=1,  # 保持特征图尺寸的填充
            bias=False
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, 
            planes, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out
    
# 定义完整的 ResNet（与官方实现一致）
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        # 初始卷积层（命名与官方一致）
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段（layer1 ~ layer4，键名与官方权重一致）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 官方权重初始化方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # 下采样调整维度（命名与官方一致）
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 实例化 ResNet50（与 torchvision.models.resnet50 完全一致）
def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],  # ResNet50 的层配置
        num_classes=num_classes
    )

def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(
        block=BasicBlock,           # 使用 BasicBlock 而非 Bottleneck
        layers=[2, 2, 2, 2],        # ResNet18 的层配置
        num_classes=num_classes,
    )

class LayerNorm2d(nn.Module):
    """二维层归一化，专为CNN设计的LayerNorm"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class ConvNeXtBlock(nn.Module):
    """ConvNeXt 基础块"""
    def __init__(self, dim: int, expansion_ratio: int = 4, drop_path_rate: float = 0.0):
        super().__init__()
        # 7x7 深度卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 层归一化
        self.norm = LayerNorm2d(dim)
        
        # 点卷积（1x1卷积）- 倒置瓶颈结构
        self.pwconv1 = nn.Linear(dim, dim * expansion_ratio)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion_ratio, dim)
        
        # 随机深度正则化
        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = StochasticDepth(drop_path_rate, mode='row')
    
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        
        # 深度卷积
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        # 层归一化和点卷积
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 转换回原始维度
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # 残差连接 + 随机深度
        return shortcut + self.drop_path(x)

class StochasticDepth(nn.Module):
    """随机深度正则化模块"""
    def __init__(self, drop_prob: float, mode: str = "row"):
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.drop_prob > 0.0:
            # 计算保留率
            survival_rate = 1.0 - self.drop_prob
            
            # 生成随机张量
            random_tensor = torch.rand(x.shape[0], *([1] * (x.dim() - 1)), 
                                      dtype=x.dtype, device=x.device)
            
            # 二值化并缩放
            random_tensor.add_(survival_rate).floor_()
            output = x / survival_rate * random_tensor
            return output
        
        return x

class ConvNeXt(nn.Module):
    """完整的ConvNeXt-Base模型，包含自定义分类头"""
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        
        # Stem层 - 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),
            LayerNorm2d(128)
        )
        
        # 各阶段设置: (通道数, 块数)
        self.stage1 = self._make_stage(128, 128, 3)
        self.stage2 = self._make_stage(128, 256, 3)
        self.stage3 = self._make_stage(256, 512, 27)
        self.stage4 = self._make_stage(512, 1024, 3)
        
        # 下采样层（过渡层）
        self.downsample1 = nn.Sequential(
            LayerNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2)
        )
        self.downsample2 = nn.Sequential(
            LayerNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=2, stride=2)
        )
        self.downsample3 = nn.Sequential(
            LayerNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2)
        )
        
        # 分类头 - 按您的要求修改
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(1024, eps=1e-6),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """创建模型的一个阶段"""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ConvNeXtBlock(out_channels))
        return nn.Sequential(*blocks)
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: Tensor) -> Tensor:
        """特征提取部分的前向传播"""
        # Stem层
        x = self.stem(x)
        
        # Stage1
        x = self.stage1(x)
        
        # Stage2 + 下采样
        x = self.downsample1(x)
        x = self.stage2(x)
        
        # Stage3 + 下采样
        x = self.downsample2(x)
        x = self.stage3(x)
        
        # Stage4 + 下采样
        x = self.downsample3(x)
        x = self.stage4(x)
        
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        """完整的前向传播"""
        # 特征提取
        x = self.forward_features(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 扁平化
        
        # 分类头
        return self.classifier(x)


# ==================== 创建模型实例 ====================
def convnext_base(num_classes: int = 23) -> ConvNeXt:
    """
    创建用于FGSC-23数据集的ConvNeXt-Base模型
    
    参数:
        num_classes: 分类类别数（FGSC-23为23类）
    
    返回:
        配置好的ConvNeXt模型
    """
    return ConvNeXt(num_classes=num_classes)

