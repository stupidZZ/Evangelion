import _init_paths
import cv2
import datetime
import argparse
import os
import sys
from common.config import config, merge_cfg_from_file

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'classification'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    # update config
    merge_cfg_from_file(args.cfg)

    args, rest = parser.parse_known_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/torch', config.TORCH_VERSION))

from functions.train import train_net

if __name__ == "__main__":
    train_net(args, config)




