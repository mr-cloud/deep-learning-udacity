
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
import urllib.request
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline
print('hello, deep learning!')

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


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


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        proxies = {'http': '127.0.0.1:8118'}
        proxy_support = urllib.request.ProxyHandler(proxies)
        opener = urllib.request.build_opener(proxy_support)
        urllib.request.install_opener(opener)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)


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
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


# train_folders = maybe_extract('notMNIST_large.tar.gz')
# test_folders = maybe_extract('notMNIST_small.tar.gz')

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

# train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
# test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']
# train_datasets = maybe_pickle(train_folders, 45000)
# test_datasets = maybe_pickle(test_folders, 1800)

## problem 2
# with open('notMNIST_small/A.pickle', 'rb') as f:
#
#     dataset = np.load(f)
#     plt.figure()
#     print('image size: (%d, %d)' % dataset[0, ::].shape)
#     plt.imshow(dataset[0,::])
#     plt.show()


## problem 3
train_folders = 'notMNIST_large'
test_folders = 'notMNIST_small'
def print_labels_stats(folders):
    files = os.listdir(folders)
    pickled_files = []
    for file in files:
        if file.__contains__('.'):
            pickled_files.append(file)

    labels_num = []
    for pickled_file in pickled_files:
        with open(os.path.join(folders, pickled_file), 'rb') as f:
            dataset = np.load(f)
            labels_num.append(dataset.__len__())

    print('\ndataset: %s\nmean of labels number: %f\nstd of lables number: %f' % (folders, np.mean(labels_num),np.std(labels_num)))
# print_labels_stats(test_folders)
# print_labels_stats(train_folders)


train_datasets = []
test_datasets = []
def get_datasets_names(folder):
    datasets = [os.path.join(folder, file) for file in os.listdir(folder) if file.__contains__('.')]
    print('datasets: ', datasets)
    return datasets
# train_datasets = get_datasets_names(train_folders)
# test_datasets = get_datasets_names(test_folders)

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

#
# train_size = 200000
# valid_size = 10000
# test_size = 10000
#
# valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
#     train_datasets, train_size, valid_size)
# _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
#
# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)
#
# def randomize(dataset, labels):
#   permutation = np.random.permutation(labels.shape[0])
#   shuffled_dataset = dataset[permutation,:,:]
#   shuffled_labels = labels[permutation]
#   return shuffled_dataset, shuffled_labels
# train_dataset, train_labels = randomize(train_dataset, train_labels)
# test_dataset, test_labels = randomize(test_dataset, test_labels)
# valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


##problem 4
# plt.figure()
# print('image size: (%d, %d)' % train_dataset[0, ::].shape)
# plt.imshow(train_dataset[0,::])
# plt.show()


# pickle_file = 'notMNIST.pickle'
#
# try:
#   f = open(pickle_file, 'wb')
#   save = {
#     'train_dataset': train_dataset,
#     'train_labels': train_labels,
#     'valid_dataset': valid_dataset,
#     'valid_labels': valid_labels,
#     'test_dataset': test_dataset,
#     'test_labels': test_labels,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', pickle_file, ':', e)
#   raise
#
#
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)


## problem 5
# import time
# import hashlib
#
# t1 = time.time()
#
# train_hashes = [hashlib.sha1(x).digest() for x in train_dataset]
# valid_hashes = [hashlib.sha1(x).digest() for x in valid_dataset]
# test_hashes  = [hashlib.sha1(x).digest() for x in test_dataset]
#
# valid_in_train = np.in1d(valid_hashes, train_hashes)
# test_in_train  = np.in1d(test_hashes,  train_hashes)
# test_in_valid  = np.in1d(test_hashes,  valid_hashes)
#
# valid_keep = ~valid_in_train
# test_keep  = ~(test_in_train | test_in_valid)
#
# valid_dataset_clean = valid_dataset[valid_keep]
# valid_labels_clean  = valid_labels [valid_keep]
#
# test_dataset_clean = test_dataset[test_keep]
# test_labels_clean  = test_labels [test_keep]
#
# t2 = time.time()
#
# print("Time: %0.2fs" % (t2 - t1))
# print("valid -> train overlap: %d samples" % valid_in_train.sum())
# print("test  -> train overlap: %d samples" % test_in_train.sum())
# print("test  -> valid overlap: %d samples" % test_in_valid.sum())
#
# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation (cleaned):', valid_dataset_clean.shape, valid_labels_clean.shape)
# print('Testing (cleaned):', test_dataset_clean.shape, test_labels_clean.shape)
#
# pickle_file = 'notMNIST_santinized.pickle'
#
# try:
#     f = open(pickle_file, 'wb')
#     save = {
#     'train_dataset': train_dataset,
#     'train_labels': train_labels,
#     'valid_dataset': valid_dataset_clean,
#     'valid_labels': valid_labels_clean,
#     'test_dataset': test_dataset_clean,
#     'test_labels': test_labels_clean,
#     }
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
#
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)

## problem 6
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


for dataset in ('notMNIST.pickle', 'notMNIST_santinized.pickle'):
    print('\n\ntraining file: ', dataset)
    with open(dataset, 'rb') as f:
        save = pickle.load(f)
        X = np.ndarray((save['train_dataset'].__len__() + save['valid_dataset'].__len__(), image_size**2), dtype=np.float32)
        y = np.concatenate((save['train_labels'], save['valid_labels']))
        print('the size of the training dataset:', X.shape)
        print('the size of the training labels:', y.shape)
        for idx, image in enumerate(save['train_dataset']):
            X[idx] = image.flatten()
        for idx, image in enumerate(save['valid_dataset']):
            X[save['train_dataset'].__len__() + idx] = image.flatten()


        test_X = np.ndarray((save['test_dataset'].__len__(), image_size**2), dtype=np.float32)
        test_y = save['test_labels']
        for idx, image in enumerate(save['test_dataset']):
            test_X[idx] = image.flatten()
        print('the size of the testing dataset:', test_X.shape)
        print('the size of testing labels:', test_y.shape)

        multi_class = 'multinomial'
        for num_examples in (50, 100, 1000, 50000, X.shape[0]):
        # for num_examples in (50,):
            print('\nnum_examples: ', num_examples)
            clf = LogisticRegression(solver='sag', multi_class=multi_class, n_jobs=-1)\
                .fit(X[:num_examples,:], y[:num_examples])

            # print the training scores
            print("training score : %.3f" % clf.score(X[:num_examples,:], y[:num_examples]))

            # print the testing results
            print("testing score : %.3f" % clf.score(test_X, test_y))
            y_true, y_pred = test_y, clf.predict(test_X)
            test_report = classification_report(y_true, y_pred)
            print('test report:\n' + test_report)


