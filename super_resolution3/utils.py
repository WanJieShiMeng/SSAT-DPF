import scipy
from scipy import ndimage
import numpy as np
import torch
import abc
import os
import random
from diffusion_model3.diffusion import Diffusion
from diffusion_model3.unet import UNetModel
import torch.nn.functional as F


# SSPSR
# https://github.com/junjun-jiang/SSPSR

def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)   # flipud  上下翻转
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)    # rot90  逆时针旋转90度*k, k = 负数时顺时针旋转
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))

# 加入噪声
class GaussianNoise:
    def __init__(self, sigma):
        # np.random.seed(seed=0)  # for reproducibility
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, img):
        img_L = (img + torch.normal(0, self.sigma, img.shape).to(self.device)).clip(0, 1)
        return img_L

# 按照信噪比加入噪声
class AddNoise:
    def __init__(self, snrdb):
        self.snr_db = snrdb

    def __call__(self, img):
        device = img.device
        # 计算信号功率
        signal_power = torch.mean(torch.abs(img)**2)

        # 计算噪声功率
        snr = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr

        # 生成高斯噪声
        noise = torch.sqrt(noise_power) * torch.randn(*img.shape)

        # 添加噪声后的信号
        noisy_img = (img + noise).clip(0, 1).to(device)
        return noisy_img

# 模糊核
def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

class AbstractBlur:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, img):
        # img_L = np.fft.ifftn(np.fft.fftn(img) * np.fft.fftn(np.expand_dims(self.k, axis=2), img.shape)).real
        img_L = ndimage.filters.convolve(img, np.expand_dims(self.kernel, axis=2), mode='wrap')
        return img_L


class GaussianBlur(AbstractBlur):
    def __init__(self, ksize=8, sigma=3):
        k = fspecial_gaussian(ksize, sigma)
        super().__init__(k)


class UniformBlur(AbstractBlur):
    def __init__(self, ksize):
        k = np.ones((ksize, ksize)) / (ksize*ksize)
        super().__init__(k)


# 模糊下采样
class AbstractDownsample(abc.ABC):
    def __init__(self, sf, kernel):
        self.sf = sf
        self.kernel = kernel


class ClassicalDownsample(AbstractDownsample):
    def __init__(self, sf, blur: AbstractBlur):
        super().__init__(sf, blur.kernel)
        self.blur = blur

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img = self.blur(img)
        img = img[0::self.sf, 0::self.sf, ...]
        return img


class GaussianDownsample(ClassicalDownsample):
    def __init__(self, sf, ksize=8, sigma=3):
        blur = GaussianBlur(ksize, sigma)
        super().__init__(sf, blur)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inference_mini_batch(model, X, T = 2000, num = 1):
    diffusion = Diffusion(T=T)
    device = X.device
    t = torch.full((num,), 10, device=device, dtype=torch.long)
    # if random:
    #     t = np.random.randint(0, T, size = (num,))
    # else:
    #     # 选择20个点
    #     t = np.linspace(0, T, num = num)
    # t = np.linspace(0, 50, num=num)
    fea_all_t = []

    for i in range(t.shape[0]):
        feature = []
        # X = torch.from_numpy(X).float()
        ti = t[i]
        ti = torch.full((1,), ti, dtype=torch.long).to(device)
        xt, tmp_noise = diffusion.forward_diffusion_sample(X, ti, device=device)
        # xt = X

        mini_batch_size = 2
        batch, c, h, w = xt.shape
        step = batch // mini_batch_size + 1

        res_feature_t_list = []
        for j in range(step):
            start = j * mini_batch_size
            end = (j+1) * mini_batch_size
            temp_xt = xt[start:end, :, :, :]
            if temp_xt.shape[0] <= 0:
                break
            noise_pred = model(temp_xt, ti, feature=True)
            temp_feature_t_list = model.return_features()
            if len(res_feature_t_list) == 0:
                res_feature_t_list = temp_feature_t_list[:]
            else:
                assert len(res_feature_t_list) == len(temp_feature_t_list)
                temp_res = []
                for j in range(len(temp_feature_t_list)):
                    temp_res.append(np.concatenate([res_feature_t_list[j], temp_feature_t_list[j]]))
                res_feature_t_list = temp_res[:]
        # for fea in res_feature_t_list:
        #     print(fea.shape)

        for fea in res_feature_t_list:
            fea = torch.from_numpy(fea)
            fea = F.interpolate(fea,size=(res_feature_t_list[-1].shape[2], res_feature_t_list[-1].shape[3]),mode='bilinear')
            if len(feature) == 0:
                feature = fea
            else:
                feature = torch.cat([feature,fea],1)
        if i == 0:
            fea_all_t = feature.unsqueeze(1)
        else:
            fea_all_t = torch.cat([fea_all_t,feature.unsqueeze(1)],1)
    fea_all_t = fea_all_t.squeeze(1).to(X.device)
    return fea_all_t



if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # 测试
    model = UNetModel(
        image_size=32,
        in_channels=128,
        model_channels=128,
        out_channels=128,
        num_res_blocks=2,
        attention_resolutions=set([4, 8]),
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        channel_mult=(1, 2, 3, 4),
        dropout=0.0,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)
    x = torch.randn(8, 128, 32, 32).to(device)
    fea = inference_mini_batch(model, x)
    print(fea.shape)
    # noise = AddNoise(30)
    # img = torch.randn(2, 128, 32, 32)
    # noisr_img = noise(img)
    # print(torch.max(noisr_img),torch.min(noisr_img))