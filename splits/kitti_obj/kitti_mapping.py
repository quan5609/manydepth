from asyncore import read
import os

import argparse
import numpy as np
import PIL.Image as pil
from tqdm import tqdm

# from utils import readlines
import csv


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def kitti_mapping():
    obj_splits = ['train', 'trainval', 'val']
    mapping_file = readlines('splits/kitti_obj/mapping/train_rand.txt')
    mapping_idx = [int(x) for x in mapping_file[0].split(',')]

    kitti_raw_mapping_file = readlines('splits/kitti_obj/mapping/train_mapping.txt')
    kitti_raw_mapping = [x.strip() for x in kitti_raw_mapping_file]

    # print(min(mapping_idx), kitti_raw_mapping[mapping_idx[0]-1])
    for split in obj_splits:
        split_dir = os.path.join('splits/kitti_obj/obj_splits', split + '.txt')
        split_file = readlines(split_dir)
        image_list = [x.strip() for x  in split_file]
        f = open(os.path.join('splits/kitti_obj', split + '.txt'), 'w')
        for image in image_list:
            line = kitti_raw_mapping[mapping_idx[int(image)]-1]
            f.write(line+ " l" + "\n")
        f.close()


    # f = open(os.path.join(path, "adaption_list.txt"), "w")
    #     image_list = [int(i.split(".")[0]) for i in os.listdir(os.path.join(path, "data")) if ".png" in i]
    #     sorted_image_list = sorted(image_list)[1:][:-1]
    #     for image_id in sorted_image_list:
    #         line = relative_path + " " + str(image_id) + " l"
    #         f.write(line+"\n")
    #     f.close()

if __name__ == "__main__":
    kitti_mapping()