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
            'axes': [1, 2],
        },
        'model': {
            'cls': 'unet_2d.Down2SkipAdd',
            'kwargs': {
                'nf': 24,
                'activation': 'torch.nn.LeakyReLU',
            },
        },

        'load_epoch': 'best',
        'checkpoint_name': 'test1',
        'infer_batch_size': 32,
    }
    return config


def prep_worker(fndict, input_keys, target_keys, new_shape):
    sample = dpr.sample_from_fndict(fndict, input_keys, target_keys)
    dpr.Prep3DUSSample.apply(sample, new_shape)
    return sample


def singles_from_split(split, prep_config):
    worker = functools.partial(
        prep_worker, input_keys=prep_config['input_keys'],
        target_keys=prep_config['target_keys'],
        new_shape=prep_config['new_shape'])
    samples = bu.map_multicore(worker, split)
    singles = [dst.Slice2DFrom3D(x, prep_config['axes']) for x in samples]
    return singles


def run_inference():
    test = load_train_valid_split()
    config = get_config()

    singles = singles_from_split(test, config['prep'])
    model = tu.build_model(**config['model'])
    tu.load_checkpoint_model(
        config['checkpoint_name'], config['load_epoch'], model)
    print(model)
    tu.get_total_params(model)
    CLOCK()

    dices = []
    for ds in singles:
        dl = dst.build_dataloader(
            ds, config['infer_batch_size'], shuffle=False)
        outputs = tu.infer_one_epoch(dl, model)
        out_0 = [x[0] for x in outputs]
        rec = ds.reconstruct_3d(out_0)
        dices.append(nu.calc_dice(rec, ds.sample.targets[0].array))
    CLOCK()
    bu.info(dices)
    print(nu.get_stats(dices))
    fu.save('dices.obj', dices)
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
