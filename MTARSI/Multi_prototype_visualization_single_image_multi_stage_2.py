# 多层特征图上采样进行原型可视化,增加强度影响

import os, cv2, torch, numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

# ────────── 1 设备 ──────────
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('>>> Using device:', device)

# ────────── 2 模型 ──────────
model_path = '/STAT/zsk/Remote_sensing_dataset/weights_files/trained_resnet50_FGSC-23_200epochs.pth'
backbone   = models.resnet50(weights=None)
backbone.fc = nn.Linear(backbone.fc.in_features, 23)
backbone.load_state_dict(torch.load(model_path, map_location=device))
backbone.eval().to(device)

# train_nodes, eval_nodes = get_graph_node_names(backbone)
# print(eval_nodes)          # 看看评估模式下有哪些节点名

# 需要抓取的层；键名任意，只要能在 return_nodes 里唯一标识
return_nodes = {
    # 'stem'   : 'stem',                      # conv-bn-relu-maxpool
    # 'relu'   : 'pre_pool',   # 112×112
    # 'layer1' : 'layer1',     # 56×56
    # 'layer2' : 'layer2',    # 28×28
    # 'layer3': 'layer3',        # 14×14
    'layer4': 'layer4',        # 7×7
}
extractor = create_feature_extractor(backbone, return_nodes=return_nodes).to(device)

# ────────── 3 输入与目标 ──────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3308, 0.3442, 0.3328],
                         std =[0.1913, 0.1849, 0.1884]),
])

img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/11/11_7_29_12347.jpg" # medical ship
# img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/11/11_7_84_11839.jpg" # medical ship1
# img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/11/11_7_28_12685.jpg" # medical ship2
# img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/1/1_3_129_10110.jpg" # aircraft-carrier
# img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/1/1_3_97_11093.jpg" # aircraft-carrier1
# img_path  = "/STAT/zsk/Remote_sensing_dataset/FGSC-23_Dataset/train/1/1_3_124_11618.jpg" # aircraft-carrier2
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/C-17/324.jpeg" # C-17
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/C-17/555.jpeg" # C-171
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/C-17/262.jpeg" # C-172
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/F-22/173.jpeg" # F-22
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/F-22/1018.jpeg" # F-221
# img_path  = "/STAT/zsk/Remote_sensing_dataset/MTARSI_Dataset/train/F-22/250.jpeg" # F-222
# img_path  = "/STAT/zsk/Remote_sensing_dataset/for_analisys_case/aircraft-carrier_and_F-22.jpg" # aircraft-carrier_and_F-22

orig_img  = Image.open(img_path).convert('RGB')
orig_np   = np.array(orig_img)
inp       = transform(orig_img).unsqueeze(0).to(device)

# ①为每一层单独写坐标+阈值 ②序数需对应
targets = {
    'layer4': dict(
        # spatial_locs=[(7, 6), (9, 5), (15, 15), (16, 16), (17, 17), (18, 16), (17, 15), (18, 14), (48, 51), (49, 50)], # layer1
        # spatial_locs=[(4, 3), (5, 3), (7, 8), (8, 8), (9, 9), (9, 8), (9, 7), (9, 7), (25, 25), (24, 25)], # layer2
        # spatial_locs=[(2, 1), (3, 2), (3, 4), (4, 4), (5, 5), (5, 4), (5, 3), (5, 3), (11, 12), (12, 12)], # layer3
        spatial_locs=[(1, 1), (2, 1), (2, 2), (3, 3), (5, 5),], # layer4
        thresholds =[0.95, 0.95, 0.95, 0.95, 0.90, 0.95, 0.95, 0.95, 0.95, 0.90],
        # weights    =[0.7, 0.6, 0.7, 1.0, 0.9, 0.8, 0.9, 0.6, 0.8, 0.6],
        weights    =[0.7, 1.0, 1.0, 0.9, 0.7], # layer4
    ), # medicalship
    # 'layer1': dict(
    #     spatial_locs=[(8, 18), (15, 20), (17, 21), (16, 18), (16, 20), (16, 22), (16, 24), (32, 33), (48, 32), (50, 30)],
    #     thresholds =[0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96],
    #     weights    =[0.5, 0.8, 1.0, 0.9, 0.9, 0.8, 0.7, 0.4, 0.4, 0.5],
    # ), # medicalship1
    # 'layer1': dict(
    #     spatial_locs=[(13, 9), (18, 18), (26, 21), (34, 39), (35, 38), (36, 38), (36, 39), (37, 38), (36, 40), (46, 50)],
    #     thresholds =[0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96],
    #     weights    =[0.5, 0.4, 0.3, 0.8, 0.9, 1.0, 0.7, 0.8, 0.9, 0.5],
    # ), # medicalship2
    # 'layer2': dict(
    #     spatial_locs=[(4, 3), (4, 3), (7, 7), (8, 8), (8, 9), (9, 8), (9, 7), (9, 8), (24, 25), (25, 24)],
    #     thresholds =[0.94, 0.94, 0.86, 0.95, 0.85, 0.92, 0.92, 0.92, 0.92, 0.92],
    # ), # medicalship
    # 'layer3': dict(
    #     spatial_locs=[(2, 2), (2, 1), (3, 3), (3, 4), (4, 4), (4, 3), (5, 4), (12, 12), (11, 12)],
    #     thresholds =[0.88, 0.90, 0.80, 0.80, 0.80, 0.80, 0.80, 0.84, 0.84],
    # ), # medicalship
    # 'layer4': dict(
    #     spatial_locs=[(1, 1), (2, 2), (3, 3), (5, 5)],
    #     thresholds =[0.93, 0.92, 0.95, 0.90],
    # ), # medicalship
        # 'pre_pool': dict(
    #     spatial_locs=[(16, 8), (16, 64), (34, 54), (38, 12), (42, 14)],
    #     thresholds =[0.98, 0.90, 0.92, 0.95, 0.97],
    # ),
    # 'layer1': dict(
    #     spatial_locs=[(18, 32), (16, 24), (49, 4), (36, 22), (7, 45), (19, 43), (41, 24), (34, 24), (34, 27), (9, 48)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.98, 0.95, 0.95, 0.95, 0.95],
    #     weights =[1.0, 1.0, 0.5, 1.0, 0.9, 0.6, 0.9, 0.8, 0.8, 0.6],
    # ), # aircraft-carrier
    # 'layer1': dict(
    #     spatial_locs=[(4, 28), (8, 36), (8, 37), (19, 29), (23, 35), (23, 38), (24, 35), (25, 34), (37, 39), (52, 16)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.8, 0.9, 0.7, 0.6, 0.7, 0.8, 1.0, 0.7, 0.9, 0.5],
    # ), # aircraft-carrier1
    # 'layer1': dict(
    #     spatial_locs=[(5, 43), (7, 48), (8, 44), (8, 46), (9, 44), (19, 27), (21, 33), (32, 27), (39, 29), (42, 24)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.8, 0.7, 1.0, 0.6, 0.6, 0.7, 0.8, 0.7, 0.5, 0.4],
    # ), # aircraft-carrier2
    # 'layer2': dict(
    #     spatial_locs=[(9, 16), (8, 12), (24, 2), (18, 11), (3, 22), (10, 20), (20, 11), (17, 12), (17, 13), (5, 24)],
    #     thresholds =[0.94, 0.94, 0.86, 0.95, 0.85, 0.92, 0.92, 0.92, 0.92, 0.92],
    # ), # aircraft-carrier
    # 'layer3': dict(
    #     spatial_locs=[(3, 8), (4, 6), (9, 3), (9, 5), (1, 11), (5, 10), (10, 5), (8, 6), (9, 6), (3, 11)],
    #     thresholds =[0.88, 0.90, 0.90, 0.80, 0.80, 0.80, 0.80, 0.84, 0.84, 0.84],
    # ), # aircraft-carrier
    # 'layer4': dict(
    #     spatial_locs=[(2, 4), (2, 3), (4, 1), (1, 5), (5, 2), (4, 3)],
    #     thresholds =[0.93, 0.92, 0.95, 0.92, 0.95, 0.90, 0.92],
    # ), # aircraft-carrier
    # 'layer1': dict(
    #     spatial_locs=[(5, 28), (8, 34), (18, 30), (25, 16), (33, 16), (36, 20), (39, 29), (38, 34), (42, 24), (44, 22)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.4, 0.5, 0.7, 0.7, 0.8, 1.0, 1.0, 0.8, 0.4, 0.7],
    # ), # C-17
    # 'layer1': dict(
    #     spatial_locs=[(2, 41), (4, 38), (6, 21), (6, 28), (9, 34), (16, 32), (16, 41), (24, 43), (32, 42), (38, 10)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.5, 0.4, 0.7, 1.0, 0.6, 0.3, 0.9, 0.8, 0.5, 0.4],
    # ), # C-171
    # 'layer1': dict(
    #     spatial_locs=[(12, 18), (18, 24), (35, 38), (35, 47), (38, 33), (41, 41), (43, 36), (44, 28), (44, 21), (48, 38)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.5, 0.4, 0.5, 1.0, 0.9, 0.3, 0.4, 0.8, 0.9, 0.4],
    # ), # C-172
    # 'layer2': dict(
    #     spatial_locs=[(3, 14), (4, 17), (9, 15), (20, 15), (22, 11), (13, 8), (19, 17), (16, 8), (18, 10)],
    #     thresholds =[0.90, 0.90, 0.85, 0.85, 0.85, 0.75, 0.85, 0.85, 0.85],
    # ), # C-17
    # 'layer3': dict(
    #     spatial_locs=[(1, 7), (2, 9), (5, 7), (8, 7), (9, 5), (6, 4), (7, 8), (8, 4), (8, 5), (7, 7)],
    #     thresholds =[0.88, 0.90, 0.90, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
    # ), # C-17
    # 'layer4': dict(
    #     spatial_locs=[(1, 4), (4, 3), (4, 2), (3, 2), (3, 3), (4, 4), (5, 3)],
    #     thresholds =[0.70, 0.80, 0.60, 0.70, 0.75, 0.80, 0.80],
    # # ), # C-17
    # 'layer1': dict(
    #     spatial_locs=[(12, 28), (12, 33), (14, 20), (14, 24), (18, 8), (19, 19), (21, 19), (26, 24), (32, 24), (42, 24)], #layer1
    #     # spatial_locs=[(6, 14), (6, 17), (7, 10), (7, 12), (9, 4), (10, 9), (10, 10), (13, 12), (16, 12), (21, 12)], #layer2
    #     # spatial_locs=[(3, 7), (3, 9), (4, 5), (5, 6), (5, 2), (5, 4), (5, 5), (7, 6), (8, 6), (11, 6)], #layer3
    #     # spatial_locs=[(2, 4), (1, 5), (2, 3), (3, 3), (3, 1), (4, 3), (5, 3)], #layer4
    #     thresholds =[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    #     weights =[0.7, 0.7, 0.5, 0.5, 0.9, 0.3, 0.4, 0.6, 1.0, 1.0],
    # ), # F-220
    # 'layer2': dict(
    #     spatial_locs=[(6, 14), (6, 17), (7, 8), (8, 14), (9, 4), (10, 9), (11, 12), (13, 14), (16, 11), (19, 12)],
    #     thresholds =[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.9, 1.0, 0.7, 0.6, 0.9, 0.3, 0.4, 0.6, 0.9, 0.7],
    # ), # F-220
    # 'layer3': dict(
    #     spatial_locs=[(5, 5), (7, 5), (7, 6), (4, 2), (4, 7), (5, 8), (8, 7), (11, 6), (3, 6), (3, 7)],
    #     thresholds =[0.88, 0.90, 0.90, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
    #     weights =[0.9, 1.0, 0.7, 0.6, 0.9, 0.3, 0.4, 0.6, 0.9, 0.7],
    # ), # F-220
    # 'layer4': dict(
    #     spatial_locs=[(2, 3), (4, 2), (4, 3), (2, 1), (2, 4), (3, 4), (1, 3), (3, 2), (2, 2)],
    #     thresholds =[0.70, 0.80, 0.60, 0.70, 0.75, 0.80, 0.80, 0.80, 0.80],
    #     weights =[0.9, 1.0, 1.0, 0.6, 0.9, 0.8, 0.4, 0.8, 1.0, 0.7],
    # ), # F-220
    # 'layer1': dict(
    #     spatial_locs=[(18, 24), (21, 26), (28, 23), (29, 32), (30, 14), (30, 39), (31, 28), (32, 33), (33, 29), (36, 22)],
    #     thresholds = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[1.0, 0.4, 0.7, 0.9, 0.4, 0.6, 0.9, 1.0, 1.0, 0.8],
    # ), # F-221
    # 'layer1': dict(
    #     spatial_locs=[(16, 18), (16, 22), (18, 22), (19, 26), (21, 42), (23, 24), (25, 22), (30, 25), (32, 26), (37, 26)],
    #     thresholds = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    #     weights =[0.4, 1.0, 0.9, 0.9, 0.4, 0.6, 0.5, 1.0, 1.0, 0.8],
    # ), # F-222
    
    # 'layer1': dict(
    #     # spatial_locs=[(18, 33), (16, 18), (47, 4), (35, 21), (7, 46), (38, 43), (41, 46), (44, 44), (48, 43), (40, 42)], # 航母
    #     # spatial_locs=[(18, 33), (16, 18), (47, 4), (35, 21), (7, 46), (17, 19), (10, 19), (17, 16), (3, 40), (17, 26)], # 飞机
    #     spatial_locs=[(38, 43), (41, 46), (44, 44), (48, 43), (40, 42), (39, 41), (37, 42), (40, 44), (47, 42), (45, 42)],
    #     thresholds =[0.95, 0.95, 0.90, 0.98, 0.98, 0.98, 0.98, 0.98, 0.95, 0.98],
    # ), # F-22 and aircraft-carrier
    # 'layer2': dict(
    #     # spatial_locs=[(9, 16), (10, 12), (22, 1), (18, 9), (4, 23), (9, 10), (5, 10), (8, 8), (2 , 20), (9, 13)], # 航母
    #     spatial_locs=[(19, 21), (21, 23), (22, 21), (20, 19), (19, 20), (20, 23), (21, 22), (20, 20)], # 飞机
    #     thresholds =[0.80, 0.80, 0.85, 0.85, 0.85, 0.75, 0.85, 0.85],
    # ), # F-22 and aircraft-carrier
    # 'layer3': dict(
    #     # spatial_locs=[(5, 7), (4, 4), (11, 0), (9, 4), (2, 11), (10, 11), (10, 12), (11, 11), (9, 11), (10, 10)],
    #     # spatial_locs=[(5, 8), (5, 6), (11, 1), (9, 5), (2, 11), (5, 5), (3, 5), (4, 4), (1 , 10), (4, 7)], # 航母
    #     spatial_locs=[(11, 10), (10, 9), (11, 11), (10, 10), (9, 11)], # 飞机
    #     thresholds =[0.80, 0.80, 0.80, 0.80, 0.80],
    # ), # F-22 and aircraft-carrier
    # 'layer4': dict(
    #     spatial_locs=[(2, 3), (2, 2), (5, 0), (4, 1), (1, 5), (5, 5)],
    #     # spatial_locs=[(2, 4), (3, 3), (5, 1), (2, 3), (4, 2), (4, 1), (1, 4)], # 航母
    #     # spatial_locs=[(5, 5)], # 飞机
    #     thresholds =[0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
    #     weights    =[1.0, 0.5, 2.0, 0.8, 0.3, 0.3],    # 这里随便举例
    # ), # F-22 and aircraft-carrier
}

# ────────── 4 前向获取多层特征 ──────────
with torch.no_grad():
    feats = extractor(inp)            # dict: {'layer1':T, 'layer2':T, ...}

# ────────── 5 生成叠加热图 ──────────
h0, w0          = orig_np.shape[:2]
combined_heat   = np.zeros((h0, w0, 3), dtype=np.float32)

for lname, cfg in targets.items():
    fmap = feats[lname][0]            # (C, H, W)
    C, H, W = fmap.shape
    fm_flat = fmap.view(C, -1)        # 预计算一次 L2-norm map
    fm_norm = fm_flat.norm(dim=0).view(H, W)

    # —— 新增：如果没写权重，则全设为1.0
    weights = cfg.get('weights', [1.0] * len(cfg['spatial_locs']))
    
    for i, ((sx, sy), thr, w) in enumerate(zip(cfg['spatial_locs'], cfg['thresholds'], weights)):
        proto_vec  = fmap[:, sx, sy]          # (C,)
        proto_norm = proto_vec.norm(p=2)
        dot_map    = (fmap * proto_vec.view(-1, 1, 1)).sum(dim=0)
        sim_map    = (dot_map / (fm_norm * proto_norm + 1e-8)).cpu().numpy()

        # 归一化 & 阈值
        sim_map -= sim_map.min()
        sim_map /= (sim_map.max() + 1e-8)
        sim_map[sim_map < thr] = 0

        # 上采样到原图
        sim_up     = cv2.resize(sim_map, (w0, h0), interpolation=cv2.INTER_LINEAR)
        # 这里就乘以你手动指定的权重
        sim_up = sim_up * w
        
        heat_uint8 = np.uint8(255 * sim_up)
        heat_rgb   = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_rgb   = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)

        combined_heat += heat_rgb.astype(np.float32)

# ────────── 6 叠加到原图并保存 ──────────
combined_heat = np.clip(combined_heat, 0, 255)
combined_heat = (combined_heat / (combined_heat.max() + 1e-8)) * 255

overlay = cv2.addWeighted(orig_np.astype(np.uint8), 0.5,
                          combined_heat.astype(np.uint8), 0.5, 0)

save_root = "/DATA/zjz/MTARSI/questionaire2.0"
os.makedirs(save_root, exist_ok=True)
base_name = os.path.splitext(os.path.basename(img_path))[0]
out_path  = os.path.join(save_root, f"{base_name}_layer4_heatmap_Medicalship_2.png")
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print('[✓] Saved to', out_path)
