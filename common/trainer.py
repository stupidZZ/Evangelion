import torch
import torch.autograd as ag
from collections import namedtuple


# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'add_step',
                            'eval_metric',
                            'locals'])


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def train(net, optimizer, lr_scheduler, train_loader, metrics, config, logger,
          batch_end_callbacks=None, epoch_end_callbacks=None):
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):

        # step LR scheduler
        lr_scheduler.step()

        # reset metrics
        metrics.reset()

        # set net to train mode
        net.train()

        # training
        for nbatch, batch in enumerate(train_loader):
            datas, labels = batch

            datas = ag.Variable(datas.cuda())
            labels = ag.Variable(labels.cuda())

            # clear the paramter gradients
            optimizer.zero_grad()

            outputs, losses = net(datas, labels)
            losses = losses.mean()
            losses.backward()
            optimizer.step()

            # update metric
            metrics.update(outputs, labels)

            # excute batch_end_callbakcs
            if batch_end_callbacks is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True,
                                                 eval_metric=metrics, locals=locals())
                _multiple_callbacks(batch_end_callbacks, batch_end_params)

        # excute epoch_end_callbacks
        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, epoch, net, optimizer)