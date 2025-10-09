import torch
from torch.autograd import Variable
import copy
import numpy as np
import math
import torch.nn.functional as F
from skimage.filters import threshold_otsu


FORWARD_BZ = 5000

device = torch.device("cuda")  

def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.to(device))
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.to(device)).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)

def generate_Gaussian_noise(image, mean=0.5, std=0.1):
    image_shape = image.shape
    noise = torch.randn(image_shape) * std + mean
    noise = torch.clamp(noise, 0.0, 1.0)
    return noise.to(device)

def create_feature_mask(image, grid_size=16):
    _, _, h, w = image.shape
    h_blocks = h // grid_size
    w_blocks = w // grid_size
    mask = torch.zeros((h, w), dtype=torch.int)
    
    feature_id = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            y_start = i * grid_size
            y_end = (i + 1) * grid_size
            x_start = j * grid_size
            x_end = (j + 1) * grid_size
            mask[y_start:y_end, x_start:x_end] = feature_id
            feature_id += 1
    return mask

def get_predicted_target(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

def get_explanation_pdt(model_type, image, model, label, exp, sg_r=None, sg_N=None, given_expl=None, binary_I=False):
    image_v = Variable(image, requires_grad=True)
    model.zero_grad()
    out = model(image_v)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])

    if exp == 'Grad':
        pdt.backward()
        grad = image_v.grad
        expl = grad.data.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'SHAP':
        expl = shap(image.cpu(), label, pdt, model, 20000)
    elif exp == 'Square':
        expl = optimal_square(image.cpu(), label, pdt, model, 20000)
    elif exp == 'NB':
        expl = optimal_nb(image.cpu(), label, pdt, model, 20000)
        if binary_I:
            expl = expl * image.cpu().numpy().flatten()
    elif exp == 'Integrated_Gradients':
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(model)
        expl = ig.attribute(image, target=label)
        expl = expl.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Int_Grad':
        for i in range(10):
            image_v = Variable(image * i / 10, requires_grad=True)
            model.zero_grad()
            out = model(image_v)
            pdt = torch.sum(out[:, label])
            pdt.backward()
            grad = image_v.grad
            if i == 0:
                expl = grad.data.cpu().numpy() / 10
            else:
                expl += grad.data.cpu().numpy() / 10
        if binary_I:
            expl = expl * image.cpu().numpy() 
    elif exp == 'Shapley_VS':
        from captum.attr import ShapleyValueSampling
        ShapVS = ShapleyValueSampling(model)
        expl = ShapVS.attribute(image, target=label, n_samples=10)
        expl = expl.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'DeepLIFT':
        from captum.attr import DeepLift
        image_temp = image.clone().requires_grad_(True)
        dl = DeepLift(model)
        expl = dl.attribute(image_temp, target=label, baselines=torch.zeros_like(image))
        expl = expl.cpu().detach().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'GradCAM':
        from captum.attr import LayerGradCam
        if model_type == 'resnet18':
            target_layer = model.layer4[1].conv2
        elif model_type == 'resnet50':
            target_layer = model.layer4[2].conv3 
        elif model_type == 'vgg16':
            target_layer = model.features[-7]
        elif model_type == 'efficientformerv2':
            target_layer = model.norm
        elif model_type == 'EfficientViT':
            target_layer = model.blocks3
        elif model_type == 'mobile_vit':
            target_layer = model.layer_5[-1]
        elif model_type == 'convnext':
            target_layer = model.features[3][-1]  
        image_temp = image.clone().requires_grad_(True)
        grad_cam = LayerGradCam(model, target_layer)
        expl = grad_cam.attribute(image_temp, target=label)
        expl = expl.repeat(1, 3, 1, 1)
        expl = F.interpolate(expl, size=(image.shape[2], image.shape[3]),mode='bilinear', align_corners=False )
        expl = expl.cpu().detach().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Guided_GradCAM':
        from captum.attr import GuidedGradCam, LayerGradCam
        image_temp = image.clone().requires_grad_(True)
        # target_layer = model.layer4[1].conv2   #resnet18
        target_layer = model.layer4[2].conv3     #resnet50
        guided_gc = GuidedGradCam(model, target_layer)
        expl = guided_gc.attribute(image_temp, target=label)
        # expl = F.interpolate(expl, size=(image.shape[2], image.shape[3]),mode='bilinear', align_corners=False )
        expl = expl.cpu().detach().numpy()
        # expl = -expl
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Lime':
        from captum.attr import Lime
        lime = Lime(model)
        feature_mask = create_feature_mask(image, grid_size=4).to(device)
        expl = lime.attribute(image, target=label, feature_mask=feature_mask, n_samples=10000, perturbations_per_eval=500)
        expl = expl.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'InputXGradient':
        from captum.attr import InputXGradient
        image_temp = image.clone().requires_grad_(True)
        IXG = InputXGradient(model)
        expl = IXG.attribute(image_temp, target=label)
        expl = expl.cpu().detach().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Saliency':
        from captum.attr import Saliency
        image_temp = image.clone().requires_grad_(True)
        saliency = Saliency(model)
        expl = saliency.attribute(image_temp, target=label)
        expl = expl.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Occlusion':
        from captum.attr import Occlusion
        occlusion = Occlusion(model)
        sliding_window_shapes = (image.shape[1], 4, 4)  # 遮挡窗口的大小
        strides = (image.shape[1], 4, 4)  # 遮挡窗口滑动的步长
        # baselines = generate_Gaussian_noise(image)  # 遮挡窗口的值设为高斯噪声
        baselines = 0
        expl = occlusion.attribute(image,
                                   target=label,
                                   sliding_window_shapes=sliding_window_shapes,
                                   strides=strides,
                                   baselines=baselines)
        expl = expl.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Smooth_Grad':
        avg_points = 50
        for count in range(int(sg_N/avg_points)):
            sample = torch.FloatTensor(sample_eps_Inf(image.cpu().numpy(), sg_r, avg_points)).to(device)
            X_noisy = image.repeat(avg_points, 1, 1, 1) + sample
            expl_eps, _ = get_explanation_pdt(X_noisy, model, label, given_expl, binary_I=binary_I)
            if count == 0:
                expl = expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])
            else:
                expl = np.concatenate([expl,
                                       expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])],
                                      axis=0)
        expl = np.mean(expl, 0)
    else:
        raise NotImplementedError('Explanation method not supported.')

    return expl, pdtr


def kernel_regression(Is, ks, ys):
    """
    *Inputs:
        I: sample of perturbation of interest, shape = (n_sample, n_feature)
        K: kernel weight
    *Return:
        expl: explanation minimizing the weighted least square
    """
    n_sample, n_feature = Is.shape
    IIk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), Is)
    Iyk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), ys)
    expl = np.matmul(np.linalg.pinv(IIk), Iyk)
    return expl

def sample_nb_Z(X, size, point):
    """
    *Inputs:
        X: flatten X vector of shape = (n_feature, )
    *Return:
        Z: perturbation of sample point
    """
    ind = np.arange(784)
    randd = np.random.normal(size=point) * 0.2 + X[ind]
    randd = np.minimum(X[ind], randd)
    randd = np.maximum(X[ind] - 1, randd)

    return randd


def sample_shap_Z(X):
    nz_ind = np.nonzero(X)[0]
    nz_ind = np.arange(X.shape[0])
    num_nz = len(nz_ind)
    bb = 0
    while bb == 0 or bb == num_nz:
        aa = np.random.rand(num_nz)
        bb = np.sum(aa > 0.5)
    sample_ind = np.where(aa > 0.5)
    Z = np.zeros(len(X))
    Z[nz_ind[sample_ind]] = 1

    return Z


def shap_kernel(Z, X):
    M = X.shape[0]
    z_ = np.count_nonzero(Z)
    return (M-1) * 1.0 / (z_ * (M - 1 - z_) * nCr(M - 1, z_))


def shap(X, label, pdt, model, n_sample):
    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)
    Xs_img = Xs.reshape(n_sample, 1, 28, 28)

    Zs = np.apply_along_axis(sample_shap_Z, 1, Xs)
    Zs_real = np.copy(Zs)
    Zs_real[Zs == 1] = Xs[Zs == 1]
    Zs_real_img = Zs_real.reshape(n_sample, 1, 28, 28)
    Zs_img = Variable(torch.tensor(Xs_img - Zs_real_img), requires_grad=False).float()
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]

    ys = pdt.data.cpu().numpy() - ys
    ks = np.apply_along_axis(shap_kernel, 1, Zs, X=X.reshape(-1))

    expl = kernel_regression(Zs, ks, ys)

    return expl


def optimal_nb(X, label, pdt, model, n_sample):
    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)
    Xs_img = Xs.reshape(n_sample, 1, 28, 28)

    Zs = np.apply_along_axis(sample_nb_Z, 1, Xs, 784, 784)
    Zs_img = Zs.reshape(n_sample, 1, 28, 28)
    Zs_img = Variable(torch.tensor(Xs_img - Zs_img), requires_grad=False).float().to(device)
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]
    ys = pdt.data.cpu().numpy() - ys

    ks = np.ones(n_sample)
    expl = kernel_regression(Zs, ks, ys)
    return expl


def optimal_square(X, label, pdt, model, n_sample):
    im_size = X.shape
    width = im_size[2]
    height = im_size[3]
    rads = np.arange(10) + 1
    n_sample = 0
    for rad in rads:
        n_sample += (width - rad + 1) * (height - rad + 1)

    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)

    Zs_img, Zs = get_imageset(Xs, im_size[1:], rads=rads)
    Zs_img = Zs_img.reshape(n_sample, 1, 28, 28)
    ks = np.ones(n_sample)

    Zs_img = Variable(torch.tensor(Zs_img), requires_grad=False).float().to(device)
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]
    ys = pdt.data.cpu().numpy() - ys

    expl = kernel_regression(Zs, ks, ys)
    return expl


def get_exp(ind, exp):
    return (exp[ind.astype(int)])


def get_imageset(image_copy, im_size, rads=[2, 3, 4, 5, 6]):
    rangelist = np.arange(np.prod(im_size)).reshape(im_size)
    width = im_size[1]
    height = im_size[2]
    ind = np.zeros(image_copy.shape)
    count = 0
    for rad in rads:
        for i in range(width - rad + 1):
            for j in range(height - rad + 1):
                ind[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 1
                image_copy[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 0
                count += 1
    return image_copy, ind

    