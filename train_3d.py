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
    run_training()
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
    return train, valid


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
        'optimizer': {
            'cls': 'torch.optim.Adam',
            'kwargs': {
                'lr': 3.0e-4,
            },
        },
        'losscalc': {
            'cls': 'torch.nn.BCELoss',
            'kwargs': {},
        },
        'scheduler': {
            'lr_patience': [10],
            'lr_reduce_coef': 0.5,
            'final_patience': 5,
        },

        'load_epoch': None,
        'checkpoint_name': 'test1',
        'n_epochs': 100,
        'valid_per': 1,
        'checkpoint_per': 100,
        'train_batch_size': 2,
        'infer_batch_size': 2,
    }
    return config


def prep_worker(fndict, input_keys, target_keys, new_shape):
    sample = dpr.sample_from_fndict(fndict, input_keys, target_keys)
    dpr.Prep3DUSSample.apply(sample, new_shape)
    return sample


def dataset_from_split(split, prep_config):
    worker = functools.partial(
        prep_worker, input_keys=prep_config['input_keys'],
        target_keys=prep_config['target_keys'],
        new_shape=prep_config['new_shape'])
    samples = bu.map_multicore(worker, split)
    singles = [dst.Vanilla3D(x) for x in samples]
    ds = dst.Container(singles)
    return ds


def train_step(epoch, dataloader, model, losscalc, optimizer, log):
    print(f'Epoch: {epoch}   training..')
    losses = tu.train_one_epoch(dataloader, model, losscalc, optimizer)
    stats = nu.get_stats(losses)
    stats['lr'] = tu.get_learning_rate(optimizer)
    print(f'Train loss = {stats["avr"]:.5f}')
    log.add_train(epoch, stats)


def valid_step(epoch, dataloader, model, losscalc, log):
    losses = tu.valid_one_epoch(dataloader, model, losscalc)
    stats = nu.get_stats(losses)
    print(f'Valid loss = {stats["avr"]:.5f}')
    log.add_valid(epoch, stats)


def run_training():
    train, valid = load_train_valid_split()
    config = get_config()

    train_ds = dataset_from_split(train, config['prep'])
    valid_ds = dataset_from_split(valid, config['prep'])

    train_dl = dst.build_dataloader(
        train_ds, config['train_batch_size'], shuffle=True, n_cpu=2)
    valid_dl = dst.build_dataloader(
        valid_ds, config['infer_batch_size'], shuffle=False, n_cpu=2)

    model, optimizer, losscalc, scheduler, log = tu.setup_training(config)
    print(model)
    print(optimizer)
    print(losscalc)
    tu.get_total_params(model)
    CLOCK()

    n_epochs = config['n_epochs']
    last = log.last_epoch()
    for epoch in range(last + 1, last + n_epochs + 1):
        train_step(epoch, train_dl, model, losscalc, optimizer, log)
        if epoch % config['valid_per'] == 0:
            valid_step(epoch, valid_dl, model, losscalc, log)
            if epoch == log.best_epoch:
                tu.save_checkpoint(
                    config['checkpoint_name'], 'best', model, optimizer,
                    scheduler, log)
            scheduler.update(epoch, log, optimizer)
            if scheduler.early_stop():
                log.add_message(epoch, 'Early stopping.')
                CLOCK()
                break
        if epoch % config['checkpoint_per'] == 0:
            tu.save_checkpoint(
                config['checkpoint_name'], epoch, model, optimizer,
                scheduler, log)
        CLOCK()
    log.display()
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
