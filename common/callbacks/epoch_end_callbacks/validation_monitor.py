import logging

class ValidationMonitor(object):
    def __init__(self, val_func, val_loader, metrics):
        super(ValidationMonitor, self).__init__()
        self.val_func = val_func
        self.val_loader = val_loader
        self.metrics = metrics

    def __call__(self, epoch_num, net, optimizer):
        self.val_func(net, self.val_loader, self.metrics)

        name, value = self.metrics.get()
        s = "Epoch[%d] \tVal-" % (epoch_num)
        for n, v in zip(name, value):
            s += "%s=%f,\t" % (n, v)

        logging.info(s)
        print(s)





