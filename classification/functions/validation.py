import torch
import torch.autograd as ag

def do_validation(net, test_loader, metrics):
    net.eval()
    metrics.reset()
    for nbatch, batch in enumerate(test_loader):
        datas, labels = batch
        datas = ag.Variable(datas.cuda())
        labels = ag.Variable(labels.cuda())

        outputs = net(datas)
        metrics.update(outputs, labels)
