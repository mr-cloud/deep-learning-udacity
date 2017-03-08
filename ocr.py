import tarfile
import numpy as np
import os
import sys
import h5py
from scipy import ndimage
import random


# 单字符30%, 64 x 64, 54 x 54, 128 x128, 95.64% at 98%
# architect: 8 convs + 3 FCL(LR, ReLU). maxout, channels: [48, 64, 128, 160], 192, 3072. dropout.
# conv: 5 x 5 zero padding, maxpooling 2 x 2, stride 2
# prediction: argmax_s(logP(S=s|H))
from scipy.misc import imresize

num_classes = 10


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    print(data_folders)
    return data_folders

maybe_extract('train.tar.gz')
maybe_extract('test.tar.gz')

image_size = 128  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
image_channels = 3
RECOGNITION_LENGTH = 5
ext_ratio = 0.3
single_digit_size = 64
crop_digit_size = 54


def prepare_datasets(mat_file):
    img_meta = h5py.File(mat_file, 'r')
    name_dataset = img_meta['digitStruct']['name']
    bbox_dataset = img_meta['digitStruct']['bbox']
    img_num = name_dataset.shape[0]
    # compute the the size of generated dataset.
    dataset_len = 0
    for idx in range(img_num):
        dataset_len += img_meta[img_meta['digitStruct']['bbox'][idx, 0]]['label'].shape[0]
    datasets = np.ndarray(shape=(dataset_len, image_size, image_size, image_channels), dtype=np.float32)
    labels = np.ndarray(shape=(dataset_len, RECOGNITION_LENGTH + 1), dtype=np.int32)  # <L, s1, s2, s3, s4, s5>, if si is absent, then set to -1.
    actual_num_imgs = 0;
    for idx in range(img_num):
        img_name = ''.join([chr(cha) for cha in img_meta[img_meta['digitStruct']['name'][idx, 0]][:, 0]])
        img_data = (ndimage.imread(img_name).astype(float) - pixel_depth / 2) / pixel_depth
        datasets[actual_num_imgs] = imresize(img_data, (image_size, image_size, image_channels))
        num_digits = img_meta[img_meta['digitStruct']['bbox'][idx, 0]]['label'].shape[0]
        example_label = [num_digits]
        for label_idx in range(num_digits):
            example_label.append(int(img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['label'][label_idx,0]][0,0]))
        labels[actual_num_imgs] = example_label.extend([-1 for _ in range(RECOGNITION_LENGTH - num_digits)])  # fill '-1' with absent digits.
        actual_num_imgs += 1
        # crop single digit to sample more examples.
        for label_idx in range(num_digits):
            label = img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['label'][label_idx,0]][0,0]
            top = img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['top'][label_idx,0]][0,0]
            left = img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['left'][label_idx,0]][0,0]
            height = img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['height'][label_idx,0]][0,0]
            width = img_meta[img_meta[img_meta['digitStruct']['bbox'][idx,0]]['width'][label_idx,0]][0,0]
            height_shift = height * ext_ratio / 2
            width_shift  = width * ext_ratio / 2
            top_start = top - height_shift if top - height_shift >= 0 else 0
            top_end = top - height_shift
            left_start = left - width_shift if left - width >=0 else 0
            left_end = left + width_shift
            single_digit = img_data[top_start: top_end, left_start: left_end, :]
            single_digit = imresize(single_digit, (single_digit_size, single_digit_size, image_channels))
            shift = random.randint(0, single_digit_size - crop_digit_size)
            single_digit = single_digit[shift:, shift:, :]
            datasets[actual_num_imgs] = imresize(single_digit, (image_size, image_size, image_channels))

            single_digit_label = [label].extend([-1 for _ in range(RECOGNITION_LENGTH - 1)])
            labels[actual_num_imgs] = single_digit_label
            actual_num_imgs += 1
    return datasets, labels

prepare_datasets('train/digitStruct.mat')