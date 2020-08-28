import numpy as np
from scipy import ndimage
from skimage.transform import resize

import re
import os
from random import SystemRandom

def parse_properties_from_path(path):
    # get base name without extension
    base_name = os.path.splitext(os.path.basename(path))[0]
    # parse properties
    property_array = re.split('-|_', base_name)
    properties = {
        'name' : property_array[0],
        'number' : property_array[1],
        'dimension' : re.split('x',re.search('[0-9]+[x][0-9]+[x][0-9]+',base_name).group(0)),
        'data_type' : property_array[-1] }
    return properties


def load_raw_image(path, endian_convert=False):
    properties = parse_properties_from_path(path)
    # read data
    data = np.fromfile(path, dtype=properties['data_type'])

    if endian_convert:
        data = data.byteswap()

    return data.reshape([int(i) for i in properties['dimension']])


def print_raw(image, data_type, path):
    with open(path, "wb") as file:
        file.write(image.astype(data_type).tobytes('Any'))


def write_raw(image, path, endian_convert=False):
    outFile = open(path, 'wb')
    if endian_convert:
        image = image.byteswap()
    outFile.write(image.tobytes())
    outFile.close()


def normalize(data, min, max):
    ptp = max - min
    nimage = (data - min) / ptp
    nimage = np.clip(nimage, 0, 1)

    return nimage.astype(np.float32)


def add_gaussian_noise(img):
    mean = 0
    var = 0.01

    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
    noise_img = img + gaussian

    return noise_img


def get_gradient_image(data):
    sx = ndimage.filters.prewitt(data.astype(float), axis=0)
    sy = ndimage.filters.prewitt(data.astype(float), axis=1)
    sz = ndimage.filters.prewitt(data.astype(float), axis=2)
    return np.sqrt(sx**2 + sy**2 + sz**2).astype(data.dtype)


def get_edge_image(data):
    inside = np.empty_like(data)
    for z in range(data.shape[0]):
        inside[z] = ndimage.binary_erosion(data[z]).astype(data.dtype)
    return data - inside


def resize_image(data, is_binary, shape, binary_threshold=0.25):

    if not is_binary:
        data_type = data.dtype
        data = resize(data.astype(float), shape)
        return data.astype(data_type)
    else:
        data_type = data.dtype
        data = resize(data.astype(float), shape) >= binary_threshold
        return data.astype(data_type)


def get_random():
    crypto = SystemRandom()
    return crypto.random()


def cutout(data):
    data_type = data.dtype

    mask = np.ones((data.shape[0], data.shape[1], data.shape[2]), np.float32)

    n_holes = 1
    # if get_random() > 0.5:
    #     n_holes = 2

    # set range to width/5 ~ width/3
    len_plane = int(data.shape[2]/5) + int(get_random() * (data.shape[2]/4 - data.shape[2]/5))
    # set range to depth/5 ~ depth/3
    len_depth = int(data.shape[0]/5) + int(get_random() * (data.shape[0]/4 - data.shape[0]/5))

    for n in range(n_holes):
        # x = np.random.randint(data.shape[2])
        # y = np.random.randint(data.shape[1])
        # z = np.random.randint(data.shape[0])
        x = int(get_random() * data.shape[2])
        y = int(get_random() * data.shape[1])
        z = int(get_random() * data.shape[0])

        x1 = np.clip(x-len_plane//2, 0, data.shape[2])
        x2 = np.clip(x+len_plane//2, 0, data.shape[2])
        y1 = np.clip(y-len_plane//2, 0, data.shape[1])
        y2 = np.clip(y+len_plane//2, 0, data.shape[1])
        z1 = np.clip(z-len_depth//2, 0, data.shape[0])
        z2 = np.clip(z+len_depth//2, 0, data.shape[0])

        mask[z1:z2, y1:y2, x1:x2] = 0.

    data = data * mask

    return data.astype(data_type)

def get_class_ids(data):
    ids = []
    for z in range(data.shape[0]):
        if np.sum(data[z]) > 0:
            ids.append(1)
        else:
            ids.append(0)
    return np.array(ids)