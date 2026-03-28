import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

from data import HSIDataLoader, TestDS, TrainDS
from unet import UNetModel
from diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img

batch_size = 128
patch_size = 16
select_spectral = []
spe = 128
channel = 1 #3d channel

epochs = 100000 # Try more!
lr = 1e-5
T=500

rgb = [70,100,36]
model_load_path = './save_model/Chikusei_1e-5'
model_name = 'unet_1180.pkl'
save_feature_path_prefix = "./save_feature/Chikusei_1e-5/save_feature"

TList = [5, 10] # , 100, 200, 400


device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_by_imgs(imgs, rgb=[1,100,199]):
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize=(12,8))
        for j in range(len(imgs)):
            plt.subplot(1,len(imgs),j+1)
            img = imgs[j][i,0,rgb,:,:]
            show_img(img)
        plt.show()

def plot_by_images_v2(imgs, rgb=[1,100,199]):
    '''
    input image shape is (spectral, height, width)
    '''
    assert len(imgs) > 0
    s,h,w = imgs[0].shape
    plt.figure(figsize=(12,8))
    for j in range(len(imgs)):
        plt.subplot(1,len(imgs),j+1)
        img = imgs[j][rgb,:,:]
        show_img(img)
    plt.show()

def plot_spectral(x0, recon_x0, num=3):
    '''
    x0, recon_x0 shape is (batch, channel, spectral, h, w)
    '''
    batch, c, s, h ,w = x0.shape
    step = h // num
    plt.figure(figsize=(20,5))
    for ii in range(num):
        i = ii * step
        x0_spectral = x0[0,0,:,i,i]
        recon_x0_spectral = recon_x0[0,0,:,i,i]
        plt.subplot(1,num,ii+1)
        plt.plot(x0_spectral, label="x0")
        plt.plot(recon_x0_spectral, label="recon")
        plt.legend()
    plt.show()


def recon_all_fig(diffusion, model, splitX, dataloader, big_img_size=[145, 145]):
    '''
    X shape is (spectral, h, w) => (batch, channel=1, 200, 145, 145)
    '''
    # 1. reconstruct
    t = torch.full((1,), diffusion.T-1, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(torch.from_numpy(splitX.astype('float32')), t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 5)

    # ---just for test---
    # recon_from_xt.append(torch.from_numpy(splitX.astype('float32')))
    # plot_by_imgs(recon_from_xt, rgb=rgb)

    # ---------

    res_xt_list = []
    for tempxt in recon_from_xt:
        big_xt = dataloader.split_to_big_image(tempxt.numpy())
        res_xt_list.append(big_xt)
    ori_data, _ = dataloader.get_ori_data()
    res_xt_list.append(ori_data)
    plot_by_images_v2(res_xt_list, rgb=rgb)

def sample_by_t(diffusion, model, X):
    num = 10
    choose_index = [3]
    x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    step = diffusion.T // num
    for ti in range(10, diffusion.T, step):
        t = torch.full((1,), ti, device=device, dtype=torch.long)
        xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
        _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 5)
        recon_x0 = recon_from_xt[-1]
        recon_from_xt.append(x0)
        print('---',ti,'---')
        plot_by_imgs(recon_from_xt, rgb=rgb)
        print("x0", x0.shape, "recon_x0", recon_x0.shape)
        plot_spectral(x0, recon_x0)

def inference_mini_batch(model, xt, t):
    mini_batch_size = 10
    batch, c, h, w = xt.shape
    step = batch // mini_batch_size + 1

    res_feature_t_list = []
    for i in range(step):
        start = i * mini_batch_size
        end = (i+1) * mini_batch_size
        temp_xt = xt[start:end, :, :, :]
        if temp_xt.shape[0] <= 0:
            break
        noise_pred = model(temp_xt, t, feature=True)
        temp_feature_t_list = model.return_features()
        if len(res_feature_t_list) == 0:
            res_feature_t_list = temp_feature_t_list[:]
        else:
            assert len(res_feature_t_list) == len(temp_feature_t_list)
            temp_res = []
            for j in range(len(temp_feature_t_list)):
                temp_res.append(np.concatenate([res_feature_t_list[j], temp_feature_t_list[j]]))
            res_feature_t_list = temp_res[:]
    for fea in res_feature_t_list:
        print(fea.shape)
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    for fea in res_feature_t_list:
        if fea.shape[-1] == 1:
            if len(feature1) == 0:
                feature1 = fea
            else:
                feature1 = np.concatenate([feature1, fea], axis=1)
        elif fea.shape[-1] == 2:
            if len(feature2) == 0:
                feature2 = fea
            else:
                feature2 = np.concatenate([feature2, fea], axis=1)
        elif fea.shape[-1] == 4:
            if len(feature3) == 0:
                feature3 = fea
            else:
                feature3 = np.concatenate([feature3, fea], axis=1)
        elif fea.shape[-1] == 8:
            if len(feature4) == 0:
                feature4 = fea
            else:
                feature4 = np.concatenate([feature4, fea], axis=1)
        else:
            if len(feature5) == 0:
                feature5 = fea
            else:
                feature5 = np.concatenate([feature5, fea], axis=1)
    return res_feature_t_list

def inference_by_t(dataloader, diffusion, model, X, ti):
    '''
    X shape is (batch, channel, spe, h, w)
    '''

    X = torch.from_numpy(X).float()
    t = torch.full((1,), ti, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(X, t, device)

    # 1. 显示调用模型直接获取隐层特征
    # noise_pred = model(xt, t, feature=True)
    # feature_t_list = model.return_features()

    feature_t_list = inference_mini_batch(model, xt, t)

    # for index, feature_matrix in enumerate(feature_t_list):
    #     path = "%s/t%s_%s.pkl" % (save_feature_path_prefix, ti, index)
    #     np.save(path, feature_matrix)
    #     print("save matrix t=%s, index=%s done." % (ti, index))
    #     # feature_matrix shape is (batch, channel, spe, h, w)
    #     fb, fc, fs, fh, fw = feature_matrix.shape
    #     temp = feature_matrix.reshape((fb,fc*fs, fh, fw)).transpose((0,2,3,1))
    #     full_feature_img = dataloader.reconstruct_image_by_light_split(temp, pathch_size=patch_size)
    #     path = "%s/t%s_%s_full.pkl" % (save_feature_path_prefix, ti, index)
    #     np.save(path, full_feature_img)
    #     print("save full matrix done. t=%s, index=%s, shape=%s" % (ti, index, str(full_feature_img.shape)))
    #
    # # 2. 对模型在该t下进行完全恢复尝试验证
    # choose_index = [3]
    # show_x0 = X[choose_index,:,:,:,:]
    # show_xt = xt[choose_index, :,:,:,:]
    # _, recon_from_xt = diffusion_model.reconstruct(model, xt=show_xt, tempT=t, num = 5) # recon_from_xt[0] shape (batch, channel, spe, h, w)
    # recon_x0 = recon_from_xt[-1]
    # recon_from_xt.append(show_x0)
    # print('---',ti,'---')
    # plot_by_imgs(recon_from_xt, rgb=rgb)
    # plot_spectral(show_x0, recon_x0)



def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 5
    step = all_size // num
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    use_t = 499
    # from xt
    t = torch.full((1,), use_t, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt, rgb=rgb)

    # from noise
    t = torch.full((1,), use_t, device=device, dtype=torch.long)

    _, recon_from_noise = diffusion.reconstruct(model, xt=x0, tempT=t, num = 10, from_noise=True, shape=x0.shape)
    plot_by_imgs(recon_from_noise, rgb=rgb)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("save model done. path=%s" % path)


def extract():
    dataloader = HSIDataLoader(
        {"data": {"data_sign": "Chikusei", "padding": False, "batch_size": batch_size, "patch_size": patch_size,
                  "select_spectral": select_spectral}})
    train_loader, X, Y = dataloader.generate_torch_dataset(light_split=True)
    diffusion = Diffusion(T=T)

    # model = SimpleUnet(_image_channels=channel)
    model = UNetModel(
        image_size=patch_size,
        in_channels=spe,
        model_channels=128,
        out_channels=spe,
        num_res_blocks=2,
        attention_resolutions=set([4, 8]),
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        channel_mult=(1, 2, 4, 8, 8),
        dropout=0.0,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    )

    assert os.path.exists(model_load_path)
    if not os.path.exists(save_feature_path_prefix):
        os.makedirs(save_feature_path_prefix)

    model_path = "%s/%s" % (model_load_path, model_name)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    print("load model done. model_path=%s" % (save_feature_path_prefix))

    for ti in TList:
        inference_by_t(dataloader, diffusion, model, X[:30], ti)
        print("feature extract t=%s done." % ti)

    print('done.')

if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    extract()
