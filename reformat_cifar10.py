import os
import cv2
import glob
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def generate_dataset(f_list, path):
    for l in f_list:
        print(l)
        l_dict = unpickle(l)
        print(l_dict.keys())

        for im_idx, im_data in enumerate(l_dict[b'data']):
            im_label = l_dict[b'labels'][im_idx]
            im_name = l_dict[b'filenames'][im_idx]

            im_label_name = label_name[im_label]
            im_data = np.reshape(im_data, [3, 32, 32])
            im_data = np.transpose(im_data, (1, 2, 0))
            mkdir('{}/{}'.format(path, im_label_name))

            cv2.imwrite('{}/{}/{}'.format(path, im_label_name, im_name.decode('utf-8')), im_data)

            # if n_type[im_idx] == 0:
            #     im_data = add_noise(im_data, noise_type='gaussian')
            # else:
            #     im_data = add_noise(im_data, noise_type='speckle')
            #
            # # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
            # # cv2.waitKey(0)
            #
            # mkdir('{}/{}'.format(path_noise, im_label_name))
            #
            # cv2.imwrite('{}/{}/{}'.format(path_noise, im_label_name, im_name.decode('utf-8')), im_data)


label_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
train_path = './dataset/TRAIN'
# train_path_noise = './dataset/TRAIN_NOISE'
test_path = './dataset/TEST'
# test_path_noise = './dataset/TEST_NOISE'

mkdir(train_path)
mkdir(test_path)
mkdir('model')
mkdir('res')

train_list = glob.glob('./dataset/data_batch_*')
test_list = glob.glob('./dataset/test_batch')

generate_dataset(train_list, train_path)
generate_dataset(test_list, test_path)
