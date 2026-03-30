# local imports

# standard library

# third party
import numpy as np


def normalize_minmax(array, eps=1e-10):
    amin, amax = array.min(), array.max()
    if np.isclose(amin, amax):
        output = np.zeros_like(array)
    else:
        output = (array - amin) / (amax - amin + eps)
    return output


def normalize_zscore(array, eps=1e-10):
    mean, std = array.mean(), array.std()
    if np.isclose(std, 0):
        output = np.zeros_like(array)
    else:
        output = (array - mean) / (std + eps)
    return output


def binarize(array, thresh=0.5):
    return (array >= thresh).astype(array.dtype)


def random_splits(container, counts, seed=None):
    if seed is not None:
        np.random.seed(seed)

    total = len(container)
    assert sum(counts) == total
    shuffled_inds = np.random.permutation(total)

    splits = []
    offset = 0
    for count in counts:
        inds = shuffled_inds[offset:offset + count]
        split = [container[i] for i in inds]
        splits.append(split)
        offset += count
    return splits


def get_stats(values):
    stats = {'min': np.min(values), 'max': np.max(values),
             'avr': np.mean(values), 'std': np.std(values)}
    return stats


def calc_dice(probas, targets, thresh=0.5):
    preds = binarize(probas, thresh)
    true_positives = (targets * preds).sum()
    total = targets.sum() + preds.sum()
    if total == 0:
        return 0
    return (2 * true_positives) / total
