import numpy as np
import os
import pprint
import shutil
import sys
import torch
import torch.optim as optim
from modules import *

from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.lib.dataset import *
from common.lib.utils.create_logger import create_logger
from common.trainer import train
from common.metrics.composite_eval_metric import CompositeEvalMetric

from validation import do_validation
import common.metrics.classification_metrics as cls_metrics

def train_net(args, config):
    np.random.seed(config.RNG_SEED)
    logger, final_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET)
    prefix = os.path.join(final_output_path, config.MODEL_PREFIX)

    # load symbol
    current_path = os.path.abspath(os.path.dirname(__file__))
    shutil.copy2(os.path.join(current_path, '../modules', config.MODULE + '.py'),
                 os.path.join(final_output_path, config.MODULE + '.py'))
    net = eval(config.MODULE + '.' + config.MODULE)(config)
    # setup multi-gpu
    gpu_num = len(config.GPUS)

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # prepare dataset
    train_set = eval(config.DATASET.DATASET)(config.DATASET.TRAIN_IMAGE_SET, config.DATASET.ROOT_PATH,
                                             config.DATASET.DATASET_PATH, config.TRAIN.SCALES)
    test_set = eval(config.DATASET.DATASET)(config.DATASET.TEST_IMAGE_SET, config.DATASET.ROOT_PATH,
                                            config.DATASET.DATASET_PATH, config.TEST.SCALES)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size = config.TRAIN.BATCH_IMAGES_PER_GPU * gpu_num,
                                               shuffle=config.TRAIN.SHUFFLE,
                                               num_workers= config.NUM_WORKER_PER_GPU * gpu_num)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size = config.TEST.BATCH_IMAGES_PER_GPU * gpu_num,
                                              shuffle=False,
                                              num_workers= config.NUM_WORKER_PER_GPU * gpu_num)

    # init parameters
    if config.TRAIN.RESUME:
        print(('continue training from ', config.TRAIN.BEGIN_EPOCH))
        # load model
        model_filename = '{}-{:04d}.model'.format(prefix, config.TRAIN.BEGIN_EPOCH-1)
        check_point = torch.load(model_filename)
        net.load_state_dict(check_point['state_dict'])
        optimizer.load_state_dict(check_point['opotimizer'])
    else:
        pass

    # setup metrices
    train_pred_names = net.get_pred_names(is_train=True)
    train_label_names = net.get_label_names(is_train=True)
    train_metrics = CompositeEvalMetric()
    train_metrics.add(cls_metrics.AccMetric(train_pred_names, train_label_names))

    val_pred_names = net.get_pred_names(is_train=False)
    val_label_names = net.get_label_names(is_train=False)
    val_metrics = CompositeEvalMetric()
    val_metrics.add(cls_metrics.AccMetric(val_pred_names, val_label_names))

    # setup callback
    batch_end_callback = [Speedometer(config.TRAIN.BATCH_IMAGES_PER_GPU * gpu_num, frequent=config.LOG_FREQUENT)]
    epoch_end_callback = [Checkpoint(os.path.join(final_output_path, config.MODEL_PREFIX)),
                          ValidationMonitor(do_validation, test_loader, val_metrics)]

    # set up optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=config.TRAIN.LR,
                          momentum=config.TRAIN.MOMENTUM,
                          weight_decay=config.TRAIN.WD,
                          nesterov=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # set up running devices
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=config.GPUS)

    # train
    train(net, optimizer=optimizer, lr_scheduler = scheduler, train_loader=train_loader,
          metrics=train_metrics, config=config, logger=logger,
          batch_end_callbacks=batch_end_callback,
          epoch_end_callbacks=epoch_end_callback)