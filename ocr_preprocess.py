import tarfile
import numpy as np
import os
import sys
import h5py
from scipy import ndimage
import random
import pickle


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

image_size = 64  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
image_channels = 3
RECOGNITION_LENGTH = 5
ext_ratio = 0.3
single_digit_size = 64
crop_digit_size = 54

last_percent_reported = None

sampling_ratio = 0.5


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def crop_image(label, top, left, height, width, img_data):
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(img_data)
    # plt.show()

    height_shift = height * ext_ratio / 2
    width_shift = width * ext_ratio / 2
    y_start = int(top - height_shift) if top - height_shift >= 0 else 0
    y_end = int(top + height + height_shift)
    x_start = int(left - width_shift) if left - width >= 0 else 0
    x_end = int(left + width + width_shift)
    single_digit = img_data[y_start: y_end, x_start: x_end, :]
    try:
        zoomed = ndimage.zoom(single_digit,
                          (single_digit_size / single_digit.shape[0], single_digit_size / single_digit.shape[1], 1))
    except ZeroDivisionError as e:
        print(e)
        return (None, None)
    if zoomed.shape != (single_digit_size, single_digit_size, image_channels):
        return (None, None)
    single_digit = zoomed
    shift = random.randint(0, single_digit_size - crop_digit_size)
    single_digit = single_digit[shift:, shift:, :]
    try:
        zoomed = ndimage.zoom(single_digit, (image_size / single_digit.shape[0], image_size / single_digit.shape[1], 1))
    except ZeroDivisionError as e:
        print(e)
        return (None, None)
    if zoomed.shape != (image_size, image_size, image_channels):
        return (None, None)

    label.extend([-1 for _ in range(RECOGNITION_LENGTH - 1)])
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(zoomed)
    # plt.show()

    return zoomed, label


def prepare_datasets(mat_file, isExtended=True):
    root = os.path.dirname(mat_file)
    img_meta = h5py.File(mat_file, 'r')
    name_dataset = img_meta['digitStruct']['name']
    bbox_dataset = img_meta['digitStruct']['bbox']
    img_num = int(name_dataset.shape[0] * sampling_ratio)
    dataset_len = img_num
    if isExtended:
        # compute the the size of generated dataset.
        for idx in range(img_num):
            dataset_len += img_meta[bbox_dataset[idx, 0]]['label'].shape[0]

    print('datasets size: ' + str((dataset_len, image_size, image_size, image_channels)))
    datasets = np.ndarray(shape=(dataset_len, image_size, image_size, image_channels), dtype=np.float32)
    labels = np.ndarray(shape=(dataset_len, RECOGNITION_LENGTH + 1), dtype=np.int32)  # <L, s1, s2, s3, s4, s5>, if si is absent, then set to -1.
    actual_num_imgs = 0
    print('Processing data for %s. This may take a while. Please wait.' % mat_file)
    for idx in range(img_num):
        download_progress_hook(idx, 1, img_num)
        num_digits = img_meta[bbox_dataset[idx, 0]]['label'].shape[0]
        if num_digits > RECOGNITION_LENGTH:
            continue
        img_name = ''.join([chr(cha) for cha in img_meta[name_dataset[idx, 0]][:, 0]])
        img_path = os.path.join(root, img_name)
        # print('processing ' + img_path)
        img_data = (ndimage.imread(img_path).astype(float) - pixel_depth / 2) / pixel_depth

        zoomed = ndimage.zoom(img_data, (image_size/img_data.shape[0], image_size/img_data.shape[1], 1))
        if zoomed.shape != (image_size, image_size, image_channels):
            continue
        datasets[actual_num_imgs] = zoomed
        example_label = [num_digits]
        # print('number of digits: ' + str(num_digits))
        if num_digits == 1:
            digit_label = int(img_meta[bbox_dataset[idx, 0]]['label'][0, 0])
            example_label.append(digit_label)
        else:
            for label_idx in range(num_digits):
                example_label.append(int(img_meta[img_meta[bbox_dataset[idx,0]]['label'][label_idx, 0]][0, 0]))
        example_label.extend([-1 for _ in range(RECOGNITION_LENGTH - num_digits)])  # fill '-1' with absent digits.
        labels[actual_num_imgs] = example_label
        actual_num_imgs += 1
        if isExtended:
            # crop single digit to sample more examples.
            if num_digits == 1:
                label = [1, int(img_meta[bbox_dataset[idx, 0]]['label'][0, 0])]
                top = img_meta[bbox_dataset[idx, 0]]['top'][0, 0]
                left = img_meta[bbox_dataset[idx, 0]]['left'][0, 0]
                height = img_meta[bbox_dataset[idx, 0]]['height'][0, 0]
                width = img_meta[bbox_dataset[idx, 0]]['width'][0, 0]
                digit_img, label = crop_image(label, top, left, height, width, img_data)
                if digit_img != None:
                    datasets[actual_num_imgs] = digit_img
                    labels[actual_num_imgs] = label
                    actual_num_imgs += 1

            else:
                for label_idx in range(num_digits):
                    label = [1, int(img_meta[img_meta[bbox_dataset[idx,0]]['label'][label_idx, 0]][0, 0])]
                    top = img_meta[img_meta[bbox_dataset[idx,0]]['top'][label_idx, 0]][0, 0]
                    left = img_meta[img_meta[bbox_dataset[idx,0]]['left'][label_idx, 0]][0, 0]
                    height = img_meta[img_meta[bbox_dataset[idx,0]]['height'][label_idx, 0]][0, 0]
                    width = img_meta[img_meta[bbox_dataset[idx,0]]['width'][label_idx, 0]][0, 0]
                    digit_img, label = crop_image(label, top, left, height, width, img_data)
                    if digit_img != None:
                        datasets[actual_num_imgs] = digit_img
                        labels[actual_num_imgs] = label
                        actual_num_imgs += 1
    datasets = datasets[:actual_num_imgs, :, :, :]
    labels = labels[:actual_num_imgs, :]
    print('actual dataset size: ' + str(datasets.shape))
    print('actual lables size: ' + str(labels.shape))
    return datasets, labels


def pickle_dataset(pickle_file, save):
    try:
      f = open(pickle_file, 'wb')
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise


train_datasets, train_labels = prepare_datasets('train/digitStruct.mat')
pickle_dataset('train.pickle', {'train_datasets': train_datasets, 'train_labels': train_labels})
del train_datasets
del train_labels
test_datasets, test_labels = prepare_datasets('test/digitStruct.mat', isExtended=False)
pickle_dataset('test.pickle', {'test_datasets': test_datasets, 'test_labels': test_labels})
