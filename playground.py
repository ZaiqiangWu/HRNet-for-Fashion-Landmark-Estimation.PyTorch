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
                        default='experiments/deepfashion2/hrnet/top1_only.yaml',
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

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    cfg.TEST.MODEL_FILE = 'models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth'
    cfg.TEST.DEEPFASHION2_BBOX_FILE = 'data/bbox_result_val.pkl'

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)  # False
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        cfg=cfg,
        target_type=cfg.MODEL.TARGET_TYPE,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    logger.info('=> Start testing...')

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


def load_model():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'play')
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    cfg.defrost()
    cfg.TEST.MODEL_FILE = 'output/deepfashion2/pose_hrnet/top1_only/2022-07-06-20-21/model_best.pth'

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)  # False
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model


def play(model, file_name):
    im = cv2.imread("./images/" + file_name + ".JPG")
    image = torch.from_numpy(im) / 255.0
    print("image shape", image.shape)
    image = image.permute([2, 0, 1])
    print("image shape", image.shape)
    mean = [0.485, 0.456, 0.406]  # [image[0].mean(),image[1].mean(),image[2].mean()]#
    std = [0.229, 0.224, 0.225]  # [image[0].std(), image[1].std(),image[2].std()]#
    # mean = [image[0].mean(), image[1].mean(), image[2].mean()]  #
    # std = [image[0].std(), image[1].std(), image[2].std()]  #
    normalize = transforms.Normalize(
        mean=mean, std=std
    )

    transform = transforms.Resize((384, 288))
    image = transform(image)

    print(image.shape)
    im = image.permute(1, 2, 0).cpu().numpy()
    im = im[:, :, [2, 1, 0]]  # convert bgr to rgb
    print(image[0].mean())
    print(image[0].std())
    image = normalize(image)
    print(image[0].mean())
    print(image[0].std())

    input = image.unsqueeze(0).cuda()
    print(input.shape)
    start = time.time()
    with torch.no_grad():
        heatmap = model(input)
    end = time.time()
    print("heatmap shape: ", heatmap.shape)
    print("Elapsed time: ", end - start)
    preds, maxvals = get_max_preds(heatmap.cpu().detach().numpy())
    print(len(preds[0]))
    print(len(maxvals[0]))
    xs = []
    ys = []
    for i in range(len(maxvals[0])):
        if maxvals[0][i] > 0.05:
            xs.append(preds[0][i][0] * 4)
            ys.append(preds[0][i][1] * 4)

    # plt.imshow(heatmap.cpu().squeeze().mean(0).detach().numpy())
    plt.imshow(im)
    print("num of landmarks", len(xs))
    scat = plt.scatter(xs, ys, c="blue")
    plt.savefig('output-' + file_name + '.png', bbox_inches='tight')
    scat.remove()
    # plt.show()


if __name__ == '__main__':
    # main()
    model = load_model()
    i = 0
    while (os.path.exists("./images/" + str(i).zfill(2) + ".JPG")):
        print("Processing: ", i)
        play(model, str(i).zfill(2))
        i = i + 1
