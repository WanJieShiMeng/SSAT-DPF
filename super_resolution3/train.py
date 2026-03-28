import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.nn import functional as F
import sys
import random
import time
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *

from dataset.HStrain import HSTrainingData
from dataset.HStest import HSTestData
from losses import HLoss
from super_resolution3.metrics import compare_mpsnr
from super_resolution3.feature_extractors import create_feature_extractor


def main():
    # parser
    parser = argparse.ArgumentParser(description="parser for HSI SISR network")
    parser.add_argument("--data_dir", type=str, default='/home/wxy/Pavia/', help="dataset directory")
    parser.add_argument("--dataset", type=str, default="Pavia", help="dataset name")
    parser.add_argument("--model", type=str, default="diffprior_group_trans_0.3_0.1", help="model_name")
    parser.add_argument("--sr_factor", type=int, default=8, help="super-resolution factor")
    parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size, default set to 16")
    parser.add_argument("--epochs", type=int, default=60, help="training epochs") # 40
    parser.add_argument("--seed", type=int, default=3407, help="start seed for model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4") # 1e-4
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    parser.add_argument("--log_interval", type=int, default=10, help="log interval for printing out info")
    parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 0)")

    parser.add_argument("--resume", type=str, help='continue training from a specific checkpoint')
    parser.add_argument("--resumeG", type=str, help='continue training from a specific checkpoint')
    parser.add_argument("--resumeD", type=str, help='continue training from a specific checkpoint')
    parser.add_argument("--use_gan", type=bool, default=False, help='continue training from a specific checkpoint')


    args = parser.parse_args()
    data_path = os.path.join(args.data_dir, args.dataset + '_x' + str(args.sr_factor))
    args.train_path = os.path.join(data_path, args.dataset + '_train')
    args.eval_path = os.path.join(data_path, args.dataset + '_eval')
    args.model_name = args.dataset + '_' + args.model + '_x' + str(args.sr_factor)
    print("Current training model: {}".format(args.model_name))
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    train(args)
    # if args.use_gan:
    #     print("===> Use GAN")
    #     train_gan(args)
    # else:
    #     train(args)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    #torch.manual_seed(args.seed)
    #if args.cuda:
   #     torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    set_random_seed(args.seed)
    print('===> Loading datasets')
    train_set = HSTrainingData(image_dir=args.train_path, augment=True) # True
    eval_set = HSTrainingData(image_dir=args.eval_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)  # pin_memory=True
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size,num_workers=4,shuffle=False)

    print('===> Building model')
    # net = diffusion_prior_net()

    net = prior_group_trans(img_size=32, n_channel=102, n_scale=args.sr_factor)
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint from '{}'".format(args.resume))
            start_epoch = int(args.resume[-6:-4]) # only support 10-99 epoch
            net.load_state_dict(torch.load(args.resume)['model'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)

    net.to(device).train()

    # Loss functions
    L1_loss = torch.nn.L1Loss()
    loss_fn = HLoss(sam_weight=0.3, gra_weight=0.1)  # 0.3, 0.1

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9,0.999))#, betas=(0.9,0.999)

    writer = SummaryWriter('runs/'+ args.model_name + '_' + str(time.ctime()).replace(":","_"))

    print('===> Start training')
    progress_bar = tqdm(total=(args.epochs - start_epoch) * len(train_set), dynamic_ncols=True)
    k = {
        "model_type": 'ddpm',
        "blocks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "input_activations": True,
        "steps": [200],
        "model_path": "/home/wxy/diffusion_super_resolution/diffusion_model3/save_model/Unet_T=1000/PaviaC/unet.pkl",
        'spe': 102,
    }
    best_epoch = 0
    best_test = 0.0
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e+1)
        losses = []
        for iteration, (x, lms, gt) in enumerate(train_loader):
            progress_bar.update(n=args.batch_size)

            x = x.to(device)
            gt = gt.to(device)
            lms = lms.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                extractor = create_feature_extractor(**k)
                feature = extractor(x)
            y = net(x,feature,lms)

            loss = loss_fn(y, gt)
            loss.backward()
            optimizer.step()

            # tensorboard logging
            losses.append(loss.item())
            if (iteration + 1) % args.log_interval == 0:
                progress_bar.set_description("===> Epoch[{}]({}/{})Loss:{:.6f}".format(e+1, iteration + 1, len(train_loader), loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss.item(), n_iter)
        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f} Learning Rate: {}".format(time.ctime(), e+1, np.mean(losses), optimizer.param_groups[0]['lr']))
        eval_loss = validate(eval_loader, net, L1_loss, device, **k) # loss修改
        test_mpsnr = test(args,net,args.sr_factor,**k)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', np.mean(losses), e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        writer.add_scalar('scalar/avg_test_mpsnr', test_mpsnr, e + 1)

        if (e + 1) % 1 == 0:
            save_checkpoint(args, net, e+1, optimizer)
        if test_mpsnr > best_test:
            best_test = test_mpsnr
            best_epoch = e + 1
    print("#" * 20)
    print("best_epoch:{} best_psnr:{}".format(best_epoch, best_test))
    print("#" * 20)



def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 160))#0.5 35
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(loader, model, criterion, device, **k):

    # switch to evaluate mode
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms = ms.to(device)
            lms = lms.to(device)
            gt = gt.to(device)
            extractor = create_feature_extractor(**k)
            # extractor_input = ms[:, torch.randint(0, ms.shape[1], (3,)), :, :]
            feature = extractor(ms)
            # feature = torch.randn(ms.shape[0], 1536, 64, 64).to(device)
            y = model(ms,feature,lms)
            # y = model(ms)
            loss = criterion(y, gt)
            losses.append(loss.item())
        print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), np.mean(losses)))
    # back to training mode
    model.train()
    return np.mean(losses)

def test(args,model,scale,**k):
    test_data_dir = '/home/wxy/'+args.dataset+'/' + args.dataset+'_x' + str(scale) + '/' + args.dataset +'_test.mat'
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    mpsnr = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, gt = ms.to(device),  gt.to(device)
            lms = lms.to(device)

            extractor = create_feature_extractor(**k)
            feature = extractor(ms)
            y = model(ms,feature,lms)
            y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
            gt = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            psnr = compare_mpsnr(x_true=gt, x_pred=y, data_range=1.)
            mpsnr.append(psnr.item())
    mpsnr = np.array(mpsnr)
    model.train()
    print("===> {}\tEpoch test Complete: Avg. MPSNR: {:.6f}".format(time.ctime(), np.mean(mpsnr)))
    return np.mean(mpsnr)


def save_checkpoint(args, model, epoch,optimizer):
    """ Save model checkpoint during training."""
    checkpoint_model_dir = 'checkpoints/' + args.model_name
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.model_name + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    checkpoint = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, ckpt_model_path)
    print("Checkpoint saved to {}".format(ckpt_model_path))


def set_random_seed(seed=3407):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
