# local imports
from . import augments as au

# standard library

# third party
import numpy as np
import torch


class Container(torch.utils.data.Dataset):
    # container for uniform single datasets

    def __init__(self, singles):
        super().__init__()
        self.singles = singles
        self.count = len(singles)
        self.each = len(singles[0])

    def __len__(self):
        return self.count * self.each

    def __getitem__(self, idx):
        general, local = divmod(idx, self.each)
        item = self.singles[general][local]
        return item


class Slice2DFrom3D(torch.utils.data.Dataset):
    # all inputs and targets must be same shape

    def __init__(self, sample, axes):
        super().__init__()
        self.sample = sample
        self.axes = axes

        self.sample_shape = sample.inputs[0].array.shape
        self.length = sum(self.sample_shape[ax] for ax in axes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inputs, targets = [], []
        current = idx
        for ax in self.axes:
            if current < self.sample_shape[ax]:
                for image in self.sample.inputs:
                    array = np.take(image.array, current, axis=ax)
                    inputs.append(array)
                for image in self.sample.targets:
                    array = np.take(image.array, current, axis=ax)
                    targets.append(array)
                break
            current -= self.sample_shape[ax]

        item = dict(inputs=inputs, targets=targets)
        return item

    def reconstruct_3d(self, outputs):
        outputs_total = sum(x.shape[0] for x in outputs)
        expected = sum(self.sample_shape[ax] for ax in self.axes)
        assert outputs_total == expected

        mapping = []
        for ax in self.axes:
            for local in range(self.sample_shape[ax]):
                mapping.append((ax, local))
        reconst = np.zeros(self.sample_shape, dtype=outputs[0].dtype)

        current = 0
        for part in outputs:
            for i in range(part.shape[0]):
                ax, local = mapping[current]
                data = part[i]
                index = [slice(None)] * 4
                index[ax] = local
                reconst[tuple(index)] += data
                current += 1

        if len(self.axes) > 1:
            reconst /= len(self.axes)
        return reconst


class RandomAugment(torch.utils.data.Dataset):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = au.apply_augments(item, self.config)
        return item


class Vanilla3D(torch.utils.data.Dataset):

    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        inputs = [x.array for x in self.sample.inputs]
        targets = [x.array for x in self.sample.targets]
        item = dict(inputs=inputs, targets=targets)
        return item


def build_dataloader(dataset, batch_size, shuffle, n_cpu=8, pin=False):
    prefetch = None if (n_cpu == 0) else 2
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_cpu,
        prefetch_factor=prefetch, pin_memory=pin)
    return dataloader
