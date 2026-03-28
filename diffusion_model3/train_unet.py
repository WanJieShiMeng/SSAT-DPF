import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from data import HSIDataLoader
from unet import UNetModel,UNetModelConnect
from diffusion_model3.diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img
plt.rcParams['font.sans-serif'] = ['Times New Roman']
batch_size = 128
patch_size = 32
select_spectral = []
spe = 102
# channel = 1 #3d channel

epochs = 10000
lr = 1e-4
T=1000

rgb = [100,30,10] # [30,50,90]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
# print(device)

def plot_by_imgs(imgs, rgb=[70,100,36]):
    '''
    选择三个通道进行图片展示
    :param imgs: 通过diffusion采样的图像,前五张展示过程,最后一章为原图像
    :param rgb:
    :return:
    '''
    assert len(imgs) > 0
    batch, c, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize=(12,8))
        for j in range(len(imgs)):
            plt.subplot(1,len(imgs),j+1)
            img = imgs[j][i,rgb,:,:]# imgs[j][i,0,rgb,:,:] 3D
            show_img(img)
            plt.axis('off')
        plt.show()


def sample_eval(diffusion, model, X):
    all_size, channel, h, w = X.shape
    num = 5 # 选5个
    step = all_size // num
    r,g,b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index,:,:,:]).float()

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


def train():
    dataloader = HSIDataLoader(
        {"data":{"data_sign":"PaviaC", "padding":False, "batch_size":batch_size, "patch_size":patch_size, "select_spectral":select_spectral}})
    train_loader,_,_ = dataloader.generate_torch_dataset(stride=True)
    diffusion = Diffusion(T=T)
    model = UNetModelConnect(
       image_size=patch_size,
       in_channels=spe,
       model_channels=128,
       out_channels=spe,
       num_res_blocks=2,
       attention_resolutions=set([4,8]),
       num_heads=4,
       num_heads_upsample=-1,
       num_head_channels=-1,
       channel_mult=(1,2,3,4),
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

    optimizer = Adam(model.parameters(), lr=lr)

    loss_metric = AvgrageMeter()
    path_prefix = "./save_model/Unet_T=1000/" + str(dataloader.data_sign) + '_p=' + str(patch_size)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    for epoch in range(epochs):
        loss_metric.reset()
        for step, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            cur_batch_size = batch.shape[0]
            t = torch.randint(0, diffusion.T , (cur_batch_size,), device=device).long()
            loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss.item(), batch.shape[0])

            if step % 20 == 0:
                print(f"[Epoch-step] {epoch} | step {step:03d} Loss: {loss.item()} ")
        print("[TRAIN EPOCH %s] loss=%s" % (epoch, loss_metric.get_avg()))

        if epoch % 1 == 0 and epoch != 0:
            path = "%s/unet_%s_%s.pkl" % (path_prefix, epoch, round(loss_metric.get_avg(),4))
            save_model(model, path)


if __name__ == "__main__":
    train()
