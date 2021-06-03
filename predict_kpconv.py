from omegaconf import DictConfig, OmegaConf
import hydra
import json
import pdal
import logging
import re

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.LAS import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

log = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="conf")
def my_app(cfg: DictConfig) -> None:

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = cfg.chosen_log  # => ModelNet40

    # Choose the index of the checkpoint to load OR None if you want to load the best checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = False

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'best_miou_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = cfg.prediction_validation_size
    config.input_threads = cfg.prediction_input_threads

    ##############
    # Prepare Data
    ##############

    log.info('')
    log.info('Data Preparation')
    log.info('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'LAS':
        test_dataset = LASDataset(config, set='test', use_potentials=True)
        test_sampler = LASSampler(test_dataset)
        collate_fn = LASCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    log.info('\nModel Preparation')
    log.info('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp, on_gpu=True)
    log.info('Done in {:.1f}s\n'.format(time.time() - t1))

    log.info('\nStart test')
    log.info('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config, num_votes=10)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)


def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


if __name__ == "__main__":
    my_app()