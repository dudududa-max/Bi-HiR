import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from lime import lime_image
from torchvision import models, transforms
from skimage.segmentation import mark_boundaries

# 加载预训练的CNN模型
model = models.vgg16(pretrained=True)
model.eval()  # 将模型设置为评估模式

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像并预处理
img_path = 'images/cat_dog.jpg'
img = Image.open(img_path).convert('RGB')
img_torch = preprocess(img)
img_array = img_torch.detach().numpy()  #将张量转化为数组形式

# 使用 np.transpose 交换轴，将形状变为 (height, width, channels)
img_array = np.transpose(img_array, (1, 2, 0))

# 定义一个预测函数，LIME将使用这个函数来获取预测结果
def predict_fn(img):
    img = np.transpose(img, (0, 3, 1, 2))  #将形状变回来
    img_tensor = torch.from_numpy(img)  #转换成张量
    with torch.no_grad():
        img_output = model(img_tensor)
        probs = F.softmax(img_output, dim=1)
    return probs.cpu().numpy()


# 使用LIME解释预测
explainer = lime_image.LimeImageExplainer()

# 选择图像的解释区域，这里我们解释整个图像
explanation = explainer.explain_instance(
    img_array,
    predict_fn,
    hide_color=0,
)

# 获取遮罩
# 假设我们关注概率最高的类别
label_to_explain = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(
    label=label_to_explain,
    positive_only=True,
    num_features=5,
    hide_rest=True
)

bool_mask = mask.astype(bool)
# 创建一个与图像大小相同的透明度遮罩
alpha_mask = np.zeros_like(img_array)
alpha_mask[bool_mask] = 1  # 对识别有贡献的区域设置为不透明

# 将原图像与透明度遮罩结合
img_with_alpha = img_array * alpha_mask

# 显示结果
plt.imshow(img_with_alpha)  # 显示带有透明度的图像
plt.axis('off')  # 不显示坐标轴
plt.show()

# #  绘制边界图
# img_Boundary = mark_boundaries(
#     img_array,
#     mask,
#     outline_color=(0, 0, 0),  # 红色轮廓
#     mode='thin'
# )
#
# # 显示结果
# plt.imshow(img_Boundary)  # 显示边界，设置透明度
# plt.axis('off')  # 不显示坐标轴
# plt.show()

