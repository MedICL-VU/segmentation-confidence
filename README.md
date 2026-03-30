Code for the ISBI 2026 paper:
Segmentation Confidence for Arbitrary CNNs

train_2d.py and infer_2d.py scripts implement training and inference with a 2D U-Net model on 3DUS placenta and MRI hippocampus images. Training includes random online augmentation.

infer_2d_with_confidence.py script implements the test-time flip augmentations and the stdev method in the paper to produce an uncertainty map and the volume-wide confidence score.

train_3d.py and infer_3d.py scripts implement training and inference with a 3D U-Net model on 3DUS placenta and MRI hippocampus images. Training doesn't include random online augmentation.

infer_3d_with_confidence.py script implements the test-time gaussian noise augmentations and the stdev+entropy method in the paper to produce an uncertainty map and the volume-wide confidence score.

Required packages:
nibabel
numpy
pandas
pynrrd
pytorch
scikit-image
scipy
