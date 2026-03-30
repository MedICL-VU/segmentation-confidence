# local imports
from . import basicutils as bu

# standard library

# third party
import nibabel
import nrrd
import numpy as np
import scipy
import skimage


def save_nifti(fn, array, header=None, affine=None):
    if header is None:
        header = nibabel.Nifti1Header()
    if affine is None:
        affine = np.eye(4)
    img = nibabel.Nifti1Image(array, header=header, affine=affine)
    nibabel.save(img, fn)


def load_nifti(fn):
    img = nibabel.load(fn)
    array = img.get_fdata()
    header = img.header
    affine = img.affine
    return array, header, affine


def nrrd_to_nifti(nrrd_header):
    affine = np.eye(4)
    spacings = np.array(nrrd_header['spacings'])
    affine[:3, :3] = np.diag(spacings)
    # LPS to RAS conversion
    affine[:3, :3] *= [-1, -1, 1]

    header = nibabel.Nifti1Header()
    header.set_data_shape(nrrd_header['sizes'])
    header.set_zooms(spacings)
    return header, affine


def load_nrrd(fn):
    array, nrrd_header = nrrd.read(fn)
    header, affine = nrrd_to_nifti(nrrd_header)
    return array, header, affine


def save_image(fn, array, header=None, affine=None):
    func_map = {
        '.nii.gz': save_nifti,
    }
    ext = bu.get_file_extension(fn, func_map.keys())
    func = func_map[ext]

    func(fn, array, header, affine)
    info = bu.get_info(array)
    print(f'[Saved] {fn} {info}')


def load_image(fn):
    func_map = {
        '.nii.gz': load_nifti,
        '.nii': load_nifti,
        '.nrrd': load_nrrd,
    }
    ext = bu.get_file_extension(fn, func_map.keys())
    func = func_map[ext]

    array, header, affine = func(fn)
    info = bu.get_info(array)
    print(f'[Loaded] {fn} {info}')
    return array, header, affine


def calculate_bbox(array):
    bbox = []
    for dim in range(array.ndim):
        axis = tuple(x for x in range(array.ndim) if x != dim)
        nonzero = np.any(array, axis=axis)
        coords = np.where(nonzero)[0]
        start, stop = coords[0], coords[-1] + 1
        bbox.append((start, stop))
    return tuple(bbox)


def crop_to_bbox(array, bbox):
    assert array.ndim == len(bbox)
    slices = tuple(slice(start, stop) for start, stop in bbox)
    return array[slices].copy()


def pad_back_bbox(array, bbox, org_shape):
    assert array.ndim == len(bbox)
    slices = tuple(slice(start, stop) for start, stop in bbox)
    padded = np.zeros(org_shape, dtype=array.dtype)
    padded[slices] = array
    return padded


def resize(array, new_shape):
    assert array.ndim == len(new_shape)
    apply_blur = np.prod(array.shape) > np.prod(new_shape)
    new = skimage.transform.resize(
        array, new_shape, order=1, mode='constant', cval=0,
        clip=True, preserve_range=True, anti_aliasing=apply_blur)
    return new


def rotate(array, degree, axes):
    new = scipy.ndimage.rotate(array, degree, axes=axes, reshape=False)
    return new
