# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from past.builtins import basestring
from common.lib.utils.collections import AttrDict
import copy
import numpy as np
import os
import os.path as osp
import yaml

__C = AttrDict()
# Consumers can get config by:
#   from core.config import cfg
config = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Common options
# ---------------------------------------------------------------------------- #
__C.TORCH_VERSION = ''
__C.RNG_SEED = 3
__C.OUTPUT_PATH = ''
__C.MODULE = ''
__C.GPUS = ()
__C.LOG_FREQUENT = 50
__C.MODEL_PREFIX = ''
__C.NUM_WORKER_PER_GPU = 4

# ---------------------- common training related params ----------------------
__C.TRAIN = AttrDict()
__C.TRAIN.SCALES = [(600, 1000)]
__C.TRAIN.LR = 0.0005
__C.TRAIN.LR_STEP = ()
__C.TRAIN.LR_FACTOR = 0.1
__C.TRAIN.WARMUP = False
__C.TRAIN.WARMUP_LR = 0.00005
__C.TRAIN.WARMUP_STEP = 1000
__C.TRAIN.WARMUP_METHOD = 'constant'
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WD = 0.0001
__C.TRAIN.BEGIN_EPOCH = 0
__C.TRAIN.END_EPOCH = 0
__C.TRAIN.RESUME = False
__C.TRAIN.FLIP= True
__C.TRAIN.SHUFFLE = True
__C.TRAIN.BATCH_IMAGES_PER_GPU = 2

# ---------------------- common testing related params ----------------------
__C.TEST = AttrDict()
__C.TEST.SCALES = [(600, 1000)]
__C.TEST.BATCH_IMAGES_PER_GPU = 1
__C.TEST.TEST_EPOCH = 0

# ---------------------- common dataset related params ----------------------
__C.DATASET = AttrDict()
__C.DATASET.DATASET = 'Cifar10'
__C.DATASET.DATASET_PATH = './data/cifar-10-batches-py'
__C.DATASET.ROOT_PATH= './data'
__C.DATASET.TRAIN_IMAGE_SET = 'cifar10_train'
__C.DATASET.TEST_IMAGE_SET = 'cifar10_test'

def namedtuple_with_defaults(typename, field_names, default_values=()):
    import collections
    """ create a namedtuple with default values """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def zip_namedtuple(nt_list):
    """ accept list of namedtuple, return a dict of zipped fields """
    if not nt_list:
        return dict()
    if not isinstance(nt_list, list):
        nt_list = [nt_list]
    for nt in nt_list:
        assert type(nt) == type(nt_list[0])
    ret = {k : [v] for k, v in nt_list[0]._asdict().items()}
    for nt in nt_list[1:]:
        for k, v in nt._asdict().items():
            ret[k].append(v)
    return ret


def update_namedtuple(config, rand_crop_samplers, rand_pad, color_jitter):
    assert isinstance(rand_crop_samplers, list)
    ret = zip_namedtuple(rand_crop_samplers)
    num_crop_sampler = len(rand_crop_samplers)
    ret['num_crop_sampler'] = num_crop_sampler  # must specify the #
    ret['rand_crop_prob'] = 1.0 / (num_crop_sampler + 1) * num_crop_sampler

    for k, v in ret.items():
        config[k] = v

    for k, v in rand_pad._asdict().items():
        config[k] = v

    for k, v in color_jitter._asdict().items():
        config[k] = v

    return config

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    (
        'FINAL_MSG',
        'MODEL.DILATION',
        'ROOT_GPU_ID',
        'RPN.ON',
        'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
        'TRAIN.DROPOUT',
        'USE_GPU_NMS',
    )
)

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'MODEL.PS_GRID_SIZE': 'RFCN.PS_GRID_SIZE',
    'MODEL.ROI_HEAD': 'FAST_RCNN.ROI_BOX_HEAD',
    'MRCNN.MASK_HEAD_NAME': 'MRCNN.ROI_MASK_HEAD',
    'TRAIN.DATASET': (
        'TRAIN.DATASETS',
        "Also convert to a tuple, e.g., " +
        "'coco_2014_train' -> ('coco_2014_train',) or " +
        "'coco_2014_train:coco_2014_valminusminival' -> " +
        "('coco_2014_train', 'coco_2014_valminusminival')"
    ),
    'TRAIN.PROPOSAL_FILE': (
        'TRAIN.PROPOSAL_FILES',
        "Also convert to a tuple, e.g., " +
        "'path/to/file' -> ('path/to/file',) or " +
        "'path/to/file1:path/to/file2' -> " +
        "('path/to/file1', 'path/to/file2')"
    ),
}


def assert_and_infer_cfg(cache_urls=True):
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if cache_urls:
        cache_cfg_urls()


def cache_cfg_urls():
    """Download URLs in the config, cache them locally, and rewrite cfg to make
    use of the locally cached file.
    """
    __C.TRAIN.WEIGHTS = cache_url(__C.TRAIN.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TEST.WEIGHTS = cache_url(__C.TEST.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TRAIN.PROPOSAL_FILES = tuple(
        [cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TRAIN.PROPOSAL_FILES]
    )
    __C.TEST.PROPOSAL_FILES = tuple(
        [cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TEST.PROPOSAL_FILES]
    )


def get_output_dir(training=True):
    """Get the output directory determined by the current global config."""
    dataset = __C.TRAIN.DATASETS if training else __C.TEST.DATASETS
    dataset = ':'.join(dataset)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        print('Deprecated config key (ignoring): {}'.format(full_key))
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a