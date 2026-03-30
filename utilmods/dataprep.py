# local imports
from . import imageutils as iu
from . import numpyutils as nu

# standard library

# third party
import numpy as np


class Image:

    def __init__(self, fn, dtype='float32'):
        self.fn = fn
        self.dtype = dtype
        self._load()
        self._validate()

    def _load(self):
        array, header, affine = iu.load_image(self.fn)
        self.array = array.astype(self.dtype)
        self.shape = array.shape
        self.header = header
        self.affine = affine

    def _validate(self):
        array = self.array.astype('float32')
        assert np.isnan(array).sum() == 0
        assert np.isinf(array).sum() == 0


class Grayscale3DUS(Image):

    def _validate(self):
        super()._validate()
        assert self.array.min() >= 0
        assert self.array.max() <= 255


class BinaryMask(Image):

    def _validate(self):
        super()._validate()
        assert np.unique(self.array).tolist() == [0, 1]


class Sample:
    # inputs and targets are lists
    # log is for reverting preprocessing

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.log = dict()


def sample_from_fndict(fndict, input_keys, target_keys):
    inputs = [Grayscale3DUS(fndict[key]) for key in input_keys]
    targets = [BinaryMask(fndict[key]) for key in target_keys]
    return Sample(inputs, targets)


class Prep3DUSSample:
    # inputs = 1 grayscale
    # targets = 0-N binary segmasks
    # modifies Sample and Image.arrays in place

    @staticmethod
    def _grayscale(image, new_shape):
        array = image.array[np.newaxis, :, :, :]
        org_shape = array.shape

        bbox = iu.calculate_bbox(array)
        array = iu.crop_to_bbox(array, bbox)
        crop_shape = array.shape

        array = iu.resize(array, new_shape)
        array = nu.normalize_minmax(array)

        image.array = array
        log = dict(org_shape=org_shape, bbox=bbox, crop_shape=crop_shape)
        return image, log

    @staticmethod
    def _segmask(image, new_shape, log):
        array = image.array[np.newaxis, :, :, :]
        assert array.shape == log['org_shape']

        array = iu.crop_to_bbox(array, log['bbox'])
        array = iu.resize(array, new_shape)
        array = nu.binarize(array)
        image.array = array
        return image

    @classmethod
    def apply(cls, sample, new_shape):
        assert len(sample.inputs) == 1
        image = sample.inputs[0]
        image, log = cls._grayscale(image, new_shape)
        sample.inputs[0] = image
        sample.log.update(log)

        for i in range(len(sample.targets)):
            image = sample.targets[i]
            sample.targets[i] = cls._segmask(image, new_shape, log)

    @staticmethod
    def revert_mask(array, log, binarize=True):
        array = iu.resize(array, log['crop_shape'])
        array = iu.pad_back_bbox(array, log['bbox'], log['org_shape'])
        if binarize:
            array = nu.binarize(array)
        return array
