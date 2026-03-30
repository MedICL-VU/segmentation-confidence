# local imports
from . import imageutils as iu
from . import numpyutils as nu

# standard library

# third party
import numpy as np
import skimage
import torch


def get_function(name):
    func_map = {
        'flip_axis': flip_axis,
        'rotate_2D': rotate_2D,
        'cutoff_2D': cutoff_2D,
        'blur': blur,
        'gamma_adjust': gamma_adjust,
        'gaussian_noise': gaussian_noise,
        'gaussian_noise_blurred': gaussian_noise_blurred,
    }
    return func_map[name]


def random():
    return torch.rand(1).item()


def randint(low, high):
    return torch.randint(low, high, (1,)).item()


def apply_augments(item, config):
    # all augments modify item in place
    for row in config:
        if random() > row['prob']:
            continue
        func = get_function(row['func'])
        func(item, **row['kwargs'])
    return item


def flip_axis(item, axis):
    # np.flip returns view, so need to create copy
    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        array = np.flip(array, axis=axis).copy()
        item['inputs'][i] = array

    for i in range(len(item['targets'])):
        array = item['targets'][i]
        array = np.flip(array, axis=axis).copy()
        item['targets'][i] = array


def rotate_2D(item, degree_range):
    low, high = degree_range
    degree = randint(low, high + 1)
    axes = (1, 2)

    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        array = iu.rotate(array, degree, axes)
        array = np.clip(array, 0, 1)
        item['inputs'][i] = array

    for i in range(len(item['targets'])):
        array = item['targets'][i]
        array = iu.rotate(array, degree, axes)
        array = nu.binarize(array)
        item['targets'][i] = array


def cutoff_2D(item):
    axis = 1 if random() < 0.5 else 2
    axlen = item['inputs'][0].shape[axis]
    cut = randint(1, axlen // 2)
    topbottom = random() < 0.5

    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if topbottom:
        src[axis] = slice(cut, None)
        dst[axis] = slice(None, -cut)
    else:
        src[axis] = slice(None, -cut)
        dst[axis] = slice(cut, None)

    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        new = np.zeros(array.shape, dtype=array.dtype)
        new[tuple(dst)] = array[tuple(src)]
        item['inputs'][i] = new

    for i in range(len(item['targets'])):
        array = item['targets'][i]
        new = np.zeros(array.shape, dtype=array.dtype)
        new[tuple(dst)] = array[tuple(src)]
        item['targets'][i] = new


def blur(item):
    sigma = random() * 0.25 + 0.50
    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        array = skimage.filters.gaussian(
            array, sigma=sigma, preserve_range=True)
        array = np.clip(array, 0, 1)
        item['inputs'][i] = array


def gamma_adjust(item):
    gamma = random() * 1.5 + 0.25
    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        array = skimage.exposure.adjust_gamma(array, gamma=gamma)
        array = np.clip(array, 0, 1)
        item['inputs'][i] = array


def gaussian_noise(item, scale, ratio):
    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        mask = torch.rand(array.shape).numpy()
        mask = (mask < ratio).astype('float32')
        noise = torch.randn(array.shape).numpy() * scale * mask
        mask = array > 0
        noise = noise * mask
        array += noise
        array = np.clip(array, 0, 1)
        item['inputs'][i] = array


def gaussian_noise_blurred(item, scale, ratio, sigma):
    for i in range(len(item['inputs'])):
        array = item['inputs'][i]
        mask = torch.rand(array.shape).numpy()
        mask = (mask < ratio).astype('float32')
        noise = torch.randn(array.shape).numpy() * scale * mask
        noise = skimage.filters.gaussian(
            noise, sigma=sigma, preserve_range=True)
        mask = array > 0
        noise = noise * mask
        array += noise
        array = np.clip(array, 0, 1)
        item['inputs'][i] = array
