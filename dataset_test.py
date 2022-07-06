# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import tools._init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_max_preds
from utils.utils import create_logger

import dataset
import models
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='experiments/deepfashion2/hrnet/dataset_test.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        logger.info("Let's use %d GPUs!" % torch.cuda.device_count())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    cfg.defrost()
    # cfg.TEST.MODEL_FILE = 'models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth'
    cfg.TEST.DEEPFASHION2_BBOX_FILE = 'data/bbox_result_val.pkl'

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])
    )
    print(len(train_dataset))
    for i in range(300):
        print("----------------------------------")
        print_info(train_dataset,i)



def print_info(datasett, index):
    input, target, target_weight, meta = datasett[index]
    print("input:", input.shape)
    print("target:", target.shape)
    print("target_weight:", target_weight.shape)
    print("meta:", meta.keys())
    print("img:", meta['image'])
    print("cat id:", meta['category_id'])
    print("roation:", meta['rotation'])
    # bgr!! need to change to rgb
    plt.imshow(input.permute(1,2,0).numpy())
    plt.show()


if __name__ == '__main__':
    main()
