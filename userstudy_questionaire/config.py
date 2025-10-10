# use the following args to obtain results on local explanations
args = {
    'seed': 0,
    'model': 'models/resnet50.pth',
    'resnet18': './models/resnet18.pth',
    'resnet50': './models/resnet50.pth',
    'vgg16': './models/vgg16.pth',
    'efficientformerv2': './models/efficientformerv2.pth',
    'EfficientViT': './models/EfficientViT.pth',
    'mobile_vit': './models/mobilevit.pth',
    # 'perts': ['Square'],
    'perts': ['Gaussian'],
    # 'exps':['Integrated_Gradients', 'Occlusion', 'DeepLIFT', 'Lime', 'InputXGradient', 'GradCAM'],
    'ques':['Integrated_Gradients', 'Occlusion', 'DeepLIFT', 'Lime', 'GradCAM'],
}

# use the following args to obtain results on global explanations
'''
args = {
    'seed': 0,
    'sen_r': 0.1,
    'sen_N': 50,  # set to 50 for the experiments used in the paper
    'sg_r': 0.2,
    'sg_N': 50,
    'model': 'models/madry_nat_tf_weight.npz',
    'perts': ['Square'],
    'exps': ['SHAP', 'Square', 'Grad', 'Int_Grad', 'GBP'],
    'sgs': ['Grad', 'Int_Grad', 'GBP']
}
'''
