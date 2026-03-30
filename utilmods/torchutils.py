# local imports
from . import basicutils as bu
from . import fileutils as fu

# standard library
import os

# third party
import torch


def build_model(cls, kwargs):
    cls = bu.dynamic_import(cls)
    model = cls(**kwargs)
    model.to(device='cuda')
    return model


def build_optimizer(model, cls, kwargs):
    cls = bu.dynamic_import(cls)
    optimizer = cls(model.parameters(), **kwargs)
    return optimizer


def build_losscalc(cls, kwargs):
    cls = bu.dynamic_import(cls)
    losscalc = cls(**kwargs)
    return losscalc


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr


def get_total_params(model, verbose=True):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    if verbose:
        print(f'Trainable params: {trainable:,}')
        print(f'Frozen    params: {frozen:,}')
        print(f'Total     params: {total:,}')
    n_params = dict(trainable=trainable, frozen=frozen, total=total)
    return n_params


def train_one_epoch(dataloader, model, losscalc, optimizer):
    model.train()
    losses = []
    for batch in dataloader:
        inputs = [x.to(device='cuda') for x in batch['inputs']]
        targets = [x.to(device='cuda') for x in batch['targets']]

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = losscalc(*outputs, *targets)
        # loss = losscalc(*outputs, *targets, model)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def valid_one_epoch(dataloader, model, losscalc):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = [x.to(device='cuda') for x in batch['inputs']]
            targets = [x.to(device='cuda') for x in batch['targets']]

            outputs = model(*inputs)
            loss = losscalc(*outputs, *targets)
            # loss = losscalc(*outputs, *targets, model)
            losses.append(loss.item())
    return losses


def infer_one_epoch(dataloader, model):
    model.eval()
    epoch_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = [x.to(device='cuda') for x in batch['inputs']]

            outputs = model(*inputs)
            outputs = [x.detach().cpu().numpy() for x in outputs]
            epoch_outputs.append(outputs)
    return epoch_outputs


class TrainingLog:

    def __init__(self):
        self.train_stats = {}
        self.valid_stats = {}
        self.messages = {}
        self.best_epoch = None
        self.best_loss = None

    def add_train(self, epoch, stats):
        self.train_stats[epoch] = stats

    def add_valid(self, epoch, stats):
        self.valid_stats[epoch] = stats
        self._update_best(epoch, stats)

    def add_message(self, epoch, text):
        if epoch in self.messages:
            self.messages[epoch] += f'\n{text}'
        else:
            self.messages[epoch] = text

    def _update_best(self, epoch, stats):
        loss = stats['avr']
        if (self.best_epoch is None) or (loss < self.best_loss):
            self.best_epoch = epoch
            self.best_loss = loss

    def last_epoch(self):
        if self.train_stats:
            return max(self.train_stats.keys())
        return 0

    def display(self):
        print('-' * 55)
        epochs = sorted(self.train_stats.keys())
        for epoch in epochs:
            train = self.train_stats[epoch]['avr']
            line = f'Epoch: {epoch}    Train = {train:.5f}'
            if epoch in self.valid_stats:
                valid = self.valid_stats[epoch]['avr']
                line += f'    Valid = {valid:.5f}'
            print(line)
            if epoch in self.messages:
                print(self.messages[epoch])

        print('-' * 55)
        epoch = self.best_epoch
        train = self.train_stats[epoch]['avr']
        valid = self.best_loss
        line = f'Best : {epoch}    Train = {train:.5f}    Valid = {valid:.5f}'
        print(line)


class PatienceScheduler:

    def __init__(self, lr_patience, lr_reduce_coef, final_patience):
        self.lr_patience = tuple(lr_patience)
        self.lr_reduce_coef = lr_reduce_coef
        self.final_patience = final_patience
        self.patience = 0

    def update(self, epoch, log, optimizer):
        self._update_patience(epoch, log)
        self._reduce_lr_if_needed(epoch, log, optimizer)

    def _update_patience(self, epoch, log):
        if epoch == log.best_epoch:
            self.patience = 0
        else:
            self.patience += 1

    def _reduce_lr_if_needed(self, epoch, log, optimizer):
        if not self.lr_patience:
            return

        if self.patience >= self.lr_patience[0]:
            old_lr = get_learning_rate(optimizer)
            new_lr = old_lr * self.lr_reduce_coef
            set_learning_rate(optimizer, new_lr)

            self.patience = 0
            self.lr_patience = self.lr_patience[1:]
            text = f'LR was {old_lr} - now reduced to {new_lr}'
            log.add_message(epoch, text)
            print(text)

    def early_stop(self):
        done_reduce = (len(self.lr_patience) == 0)
        done_final = (self.patience >= self.final_patience)
        return done_reduce and done_final


def get_checkpoint_fn(name, epoch):
    fn = f'{name}_ep_{epoch}.obj'
    fn = os.path.join('.', 'checkpoints', fn)
    return fn


def save_checkpoint(name, epoch, model, optimizer, scheduler, log):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
        'log': log,
    }
    fn = get_checkpoint_fn(name, epoch)
    fu.save(fn, checkpoint)


def load_checkpoint(name, epoch, model, optimizer):
    # modifies model and optimizer in place
    fn = get_checkpoint_fn(name, epoch)
    checkpoint = fu.load(fn)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = checkpoint['scheduler']
    log = checkpoint['log']
    return scheduler, log


def load_checkpoint_model(name, epoch, model):
    # modifies model in place
    fn = get_checkpoint_fn(name, epoch)
    checkpoint = fu.load(fn)
    model.load_state_dict(checkpoint['model'])


def load_checkpoint_log(name, epoch):
    fn = get_checkpoint_fn(name, epoch)
    checkpoint = fu.load(fn)
    return checkpoint['log']


def setup_training(config):
    model = build_model(**config['model'])
    optimizer = build_optimizer(model, **config['optimizer'])
    losscalc = build_losscalc(**config['losscalc'])

    if config['load_epoch'] is None:
        scheduler = PatienceScheduler(**config['scheduler'])
        log = TrainingLog()
    else:
        scheduler, log = load_checkpoint(
            config['checkpoint_name'], config['load_epoch'], model, optimizer)
        log.display()
    return model, optimizer, losscalc, scheduler, log
