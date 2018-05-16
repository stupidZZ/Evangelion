import torch


class Checkpoint(object):
    def __init__(self, prefix):
        super(Checkpoint, self).__init__()
        self.prefix = prefix

    def __call__(self, epoch_num, net, optimizer):
        param_name = '{}-{:04d}.model'.format(self.prefix, epoch_num)
        checkpoint_dict = {}
        checkpoint_dict['state_dict'] = net.state_dict()
        checkpoint_dict['opotimizer'] = optimizer.state_dict()
        torch.save(checkpoint_dict, param_name)



