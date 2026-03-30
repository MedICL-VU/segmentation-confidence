# local imports
import utilmods.basicutils as bu
import utilmods.dataprep as dpr
import utilmods.datasets as dst
import utilmods.fileutils as fu
import utilmods.numpyutils as nu
import utilmods.torchutils as tu

# standard library
import functools
import pdb
import sys
import traceback

# third party
import numpy as np


def main():
    # temp()
    run_inference()
    pass


def temp():
    pdb.set_trace()
    pass


def load_catalog():
    base = '/media/baris/Data/VandyPenn/PlacentaProject/datasets/catalogs/'
    fn = base + 'df_v1_qc_210_catalog.csv'
    df = fu.load(fn)
    catalog = df.to_dict('records')
    return catalog


def load_train_valid_split():
    cat = load_catalog()
    assert len(cat) == 121
    train, valid, test = nu.random_splits(cat, [40, 40, 41], seed=13)
    return test


def get_config():
    config = {
        'prep': {
            'input_keys': ['grayscale'],
            'target_keys': ['placenta'],
            'new_shape': [1, 128, 128, 128],
        },
        'model': {
            'cls': 'unet_3d.Down2SkipAdd',
            'kwargs': {
                'nf': 32,
                'activation': 'torch.nn.LeakyReLU',
            },
        },

        'load_epoch': 'best',
        'checkpoint_name': 'test1',
        'infer_batch_size': 1,
    }
    return config


def prep_worker(fndict, input_keys, target_keys, new_shape):
    sample = dpr.sample_from_fndict(fndict, input_keys, target_keys)
    dpr.Prep3DUSSample.apply(sample, new_shape)
    return sample


def samples_from_split(split, prep_config):
    worker = functools.partial(
        prep_worker, input_keys=prep_config['input_keys'],
        target_keys=prep_config['target_keys'],
        new_shape=prep_config['new_shape'])
    samples = bu.map_multicore(worker, split)
    return samples


def calc_vvc(array):
    vols = array.sum(axis=(1, 2, 3, 4))
    vvc = vols.std() / vols.mean()
    return vvc


def calc_uncertainty_stdev(array):
    uncertainty = array.std(axis=0)
    uncertainty = nu.normalize_minmax(uncertainty)
    return uncertainty


def calc_uncertainty_entropy(array):
    eps = 1e-6
    part1 = -array * np.log(array + eps)
    part2 = (1 - array) * np.log(1 - array + eps)
    entropy = part1 - part2
    entropy = entropy.mean(axis=0)
    entropy = nu.normalize_minmax(entropy)
    return entropy


def calc_uncertainty_mix(array):
    stdev = calc_uncertainty_stdev(array)
    entropy = calc_uncertainty_entropy(array)
    uncertainty = (stdev + entropy) / 2
    return uncertainty


def calc_confidence(probas, uncertainty):
    totals = []
    for i in range(1, 11):
        mask = (uncertainty < (i / 10)).astype('float32')
        total = (mask * probas).sum()
        totals.append(total)
    assert totals == sorted(totals)

    ratios = []
    for i in range(9):
        ratios.append(totals[i] / totals[9])
    for i in range(8):
        ratios.append(totals[i] / totals[i + 1])

    confidence = np.mean(ratios)
    return confidence


def infer_step(ds, model, batchsize):
    dl = dst.build_dataloader(ds, batchsize, shuffle=False)
    outputs = tu.infer_one_epoch(dl, model)
    out_0 = outputs[0][0][0]
    return out_0


def infer_with_confidence(sample, model, batchsize, repeat):
    shape = (repeat + 1,) + sample.inputs[0].array.shape
    probas = np.zeros(shape, dtype='float32')

    ds = dst.Vanilla3D(sample)
    probas[0] = infer_step(ds, model, batchsize)

    augment = [
        {'func': 'gaussian_noise_blurred',
         'prob': 1.0,
         'kwargs': {'scale': 0.5, 'ratio': 0.2, 'sigma': 1.0}}
    ]
    ds = dst.RandomAugment(ds, augment)
    for i in range(1, repeat + 1):
        probas[i] = infer_step(ds, model, batchsize)
    return probas


def run_inference():
    test = load_train_valid_split()
    config = get_config()

    samples = samples_from_split(test, config['prep'])
    model = tu.build_model(**config['model'])
    tu.load_checkpoint_model(
        config['checkpoint_name'], config['load_epoch'], model)
    print(model)
    tu.get_total_params(model)
    CLOCK()

    dices, confidences, vvcs = [], [], []
    for sample in samples:
        probas = infer_with_confidence(
            sample, model, config['infer_batch_size'], repeat=15)

        vvcs.append(calc_vvc(probas))
        uncertainty = calc_uncertainty_mix(probas)
        avr = probas.mean(axis=0)
        conf = calc_confidence(avr, uncertainty)
        confidences.append(conf)
        dices.append(nu.calc_dice(avr, sample.targets[0].array))
    CLOCK()
    bu.info(dices, confidences, vvcs)
    print(nu.get_stats(dices))
    print(nu.get_stats(confidences))
    print(nu.get_stats(vvcs))
    fu.save('dices.obj', dices)
    fu.save('confidences.obj', confidences)
    fu.save('vvcs.obj', vvcs)
    # pdb.set_trace()


if __name__ == '__main__':
    CLOCK = bu.Clock()
    try:
        main()
        CLOCK()
        bu.beep('done')
    except Exception:
        CLOCK()
        bu.beep('error')
        print('-' * 66)
        traceback.print_exc()
        print('-' * 66)
        pdb.post_mortem()
    finally:
        sys.exit()
