import argparse
import copy
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch.utils.data import DataLoader

from models import *
from metrics import quality_assessment
from dataset import HSTestData
import os

from torch.nn.functional import interpolate
from super_resolution3.feature_extractors import create_feature_extractor
# global settings

data_name = 'Pavia'
scale = 4
test_data_dir = '/home/wxy/' + data_name + '/' + data_name +'_x'+str(scale)+'/' + data_name +'_test.mat'
model_name = 'overlapping'
save_model_title = model_name + '_' + data_name +'_x' + str(scale)

save_path = ".pth"
result_dir = 'result/' + save_model_title + '.mat'
k = {
    "model_type": 'ddpm',
    "blocks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "input_activations": True,
    "steps": [200],
    "model_path": "/home/wxy/diffusion_super_resolution/diffusion_model3/save_model/Unet_T=1000/PaviaC/unet.pkl",
    'spe': 102,
}

def main():
    # parsers
    parser = argparse.ArgumentParser(description="parser for HyperSR network")
    parser.add_argument("--cuda", type=int, required=False, default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")
    parser.add_argument("--use_gan", type=bool, default=False, help='continue training from a specific checkpoint')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    test(args)


def test(args):
    # set_random_seed(3407)
    device = torch.device("cuda:0" if args.cuda else'cpu')
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    logger = pd.DataFrame()
    with torch.no_grad():

        model = prior_group_trans(n_subs=8,n_ovls=6,img_size=32, n_channel=102, n_scale=scale)

        model.load_state_dict(torch.load(save_path)['model'])
        model.to(device).eval()
        result = []
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
            indices = quality_assessment(gt, y, data_range=1., ratio=8)
            logger = logger._append([indices], ignore_index=True)
            result.append(y)

    print("===> {}\t Testing RGB Complete: Avg. SAM: {:.4f}, Avg. MPSNR: {:.4f}, Avg. MSSIM: {:.4f}".format(
            time.ctime(), logger['SAM'].mean(), logger['MPSNR'].mean(), logger['MSSIM'].mean()))



def set_random_seed(seed=3407):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    main()
