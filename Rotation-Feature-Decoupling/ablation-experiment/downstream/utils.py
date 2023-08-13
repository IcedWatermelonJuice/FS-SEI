import os
from shutil import copyfile
import torch
import torch.nn as nn
import datetime
import logging


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


def set_log_file_handler(log_dir):
    logging.getLogger().handlers = []
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    strHandler = logging.StreamHandler()
    strHandler.setFormatter(formatter)
    logger.addHandler(strHandler)
    logger.setLevel(logging.INFO)

    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)
    now_str = datetime.datetime.now().strftime('%m%d_%H%M%S')
    log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.log')
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    return logger


def get_pretrain_encoder(network, logger, pretrained_path, process_unit="cuda"):
    logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))
    assert (os.path.isfile(pretrained_path))
    if "cpu" == process_unit:
        pretrained_model = torch.load(pretrained_path, map_location='cpu')
    elif "gpu" == process_unit or "cuda" == process_unit:
        pretrained_model = torch.load(pretrained_path)
    else:
        raise ValueError('Process unit platform is not specified')

    if 'module' in list(pretrained_model['network'].keys())[0]:
        logger.info('==> Network keys in pre-trained file %s contain \"module\"' % (pretrained_path))
        from collections import OrderedDict
        pretrained_model_nomodule = OrderedDict()
        for key, value in pretrained_model['network'].items():
            key_nomodule = key[7:]  # remove module
            pretrained_model_nomodule[key_nomodule] = value
    else:
        pretrained_model_nomodule = pretrained_model['network']

    if pretrained_model_nomodule.keys() == network.state_dict().keys():
        network.load_state_dict(pretrained_model_nomodule)
    else:
        logger.info('==> WARNING: network parameters in pre-trained file %s do not strictly match' % (pretrained_path))
        network.load_state_dict(pretrained_model_nomodule, strict=False)
    return network
