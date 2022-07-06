#!/bin/bash
python tools/train.py \
    --cfg experiments/deepfashion2/hrnet/top_only.yaml \
     MODEL.PRETRAINED models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth