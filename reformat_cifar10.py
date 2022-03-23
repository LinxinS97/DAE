import os
import cv2
import glob
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def add_noise(img, noise_type="gaussian"):
    row, col, c = 32, 32, 3
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        var = 10
        sigma = var ** .5

        noise = np.random.normal(mean, sigma, img.shape)
        noise = noise.reshape((row, col, c))
        img = img + noise
        return img

    if noise_type == "speckle":
        noise = np.random.randn(row, col, c)
        img = img + img * noise
        return img

def generate_dataset(f_list, path_clean, path_noise):
    for l in f_list:
        print(l)
        l_dict = unpickle(l)
        print(l_dict.keys())
        n_type = np.random.randint(2, size=len(l_dict[b'labels']))

        for im_idx, im_data in enumerate(l_dict[b'data']):
            im_label = l_dict[b'labels'][im_idx]
            im_name = l_dict[b'filenames'][im_idx]

            im_label_name = label_name[im_label]
            im_data = np.reshape(im_data, [3, 32, 32])
            im_data = np.transpose(im_data, (1, 2, 0))
            if not os.path.exists("{}/{}".format(path_clean, im_label_name)):
                os.mkdir("{}/{}".format(path_clean, im_label_name))

            cv2.imwrite("{}/{}/{}".format(path_clean, im_label_name, im_name.decode('utf-8')), im_data)

            if n_type[im_idx] == 0:
                im_data = add_noise(im_data, noise_type='gaussian')
            else:
                im_data = add_noise(im_data, noise_type='speckle')

            # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
            # cv2.waitKey(0)

            if not os.path.exists("{}/{}".format(path_noise, im_label_name)):
                os.mkdir("{}/{}".format(path_noise, im_label_name))

            cv2.imwrite("{}/{}/{}".format(path_noise, im_label_name, im_name.decode('utf-8')), im_data)


label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

train_list = glob.glob("./dataset/data_batch_*")
test_list = glob.glob("./dataset/test_batch")

train_path_clean = "./dataset/TRAIN"
train_path_noise = './dataset/TRAIN_NOISE'
test_path_clean = "./dataset/TEST"
test_path_noise = "./dataset/TEST_NOISE"

generate_dataset(train_list, train_path_clean, train_path_noise)
generate_dataset(test_list, test_path_clean, test_path_noise)
