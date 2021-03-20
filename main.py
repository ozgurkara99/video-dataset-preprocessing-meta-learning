import argparse
from loader import DATALOADER
import os
from util import split_sets
import torch

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="dataset", help="validation set list")
    parser.add_argument("--val", type=str, default="smsm-100/val.list", help="validation set list")
    parser.add_argument("--test", type=str, default="smsm-100/test.list", help="test set list")
    parser.add_argument("--train", type=str, default="smsm-100/train.list", help="training set list")
    parser.add_argument("--src", type=str, default="something v2/20bn-something-something-v2/", help="Directory to which the videos will be copied")
    parser.add_argument("--k", type=int, default=1, help="k-shot")
    parser.add_argument("--n", type=int, default=5, help="n-way")
    parser.add_argument("--T", type=int, default=8, help="Number of frames for video split")
    opt = parser.parse_args()

    return opt


opt = get_args()
print(opt)
if opt.target in os.listdir():
    print("Data were splitted before.")
else:
    print("Data is being splitted into 3 parts: Train, Validation, Test...")
    split_sets(opt)
train = DATALOADER(opt, "train")
query_x, query_y, support_x, support_y = train.random_sample_each_episode()
print("Size of Query set:", query_x.size(), "Size of support set:", support_x.size())