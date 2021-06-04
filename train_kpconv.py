from omegaconf import DictConfig, OmegaConf
import hydra
import json
import pdal
import logging
import re

# Common libs
import signal
import os
import sys

# Dataset
from datasets.LAS import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="conf")
def my_app(cfg: DictConfig) -> None:

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

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'best_miou_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    log.info('')
    log.info('Data Preparation')
    log.info('****************')

    # Initialize configuration class
    config = LASConfig(cfg)
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets
    log.info(config.path)
    training_dataset = LASDataset(config, set='training', use_potentials=True)
    test_dataset = LASDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = LASSampler(training_dataset)
    test_sampler = LASSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=LASCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=LASCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    log.info('\nModel Preparation')
    log.info('*****************')

    # Define network model
    t1 = time.time()
    label_value_ids = np.array([training_dataset.label_to_idx[l] for l in training_dataset.label_values])
    net = KPFCNN(config, label_value_ids, training_dataset.ignored_labels)

    debug = False
    if debug:
        log.info('\n*************************************\n')
        log.info(net)
        log.info('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                log.info(param.shape)
        log.info('\n*************************************\n')
        log.info("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        log.info('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    log.info('Done in {:.1f}s\n'.format(time.time() - t1))

    log.info('\nStart training')
    log.info('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    log.info('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)


class LASConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    def __init__(self, cfg):
        super(LASConfig, self).__init__()

        # Number of CPU threads for the input pipeline
        self.input_threads = cfg.input_threads

        ###################
        # KPConv parameters
        ###################

        # Radius of the input sphere
        self.in_radius = cfg.in_radius

        # Number of kernel points
        self.num_kernel_points = cfg.num_kernel_points

        # Size of the first subsampling grid in meter
        self.first_subsampling_dl = cfg.first_subsampling_dl

        # Radius of convolution in "number grid cell". (2.5 is the standard value)
        self.conv_radius = cfg.conv_radius

        # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
        self.deform_radius = cfg.deform_radius

        # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
        self.KP_extent = cfg.KP_extent

        # Behavior of convolutions in ('constant', 'linear', 'gaussian')
        self.KP_influence = cfg.KP_influence

        # Aggregation function of KPConv in ('closest', 'sum')
        self.aggregation_mode = cfg.aggregation_mode

        # Choice of input features
        self.first_features_dim = cfg.first_features_dim
        self.in_features_dim = cfg.in_features_dim

        # Can the network learn modulations
        self.modulated = cfg.modulated

        # Batch normalization parameters
        self.use_batch_norm = cfg.use_batch_norm
        self.batch_norm_momentum = cfg.batch_norm_momentum

        # Deformable offset loss
        # 'point2point' fitting geometry by penalizing distance from deform point to input points
        # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
        self.deform_fitting_mode = cfg.deform_fitting_mode
        self.deform_fitting_power = cfg.deform_fitting_power              # Multiplier for the fitting/repulsive loss
        self.deform_lr_factor = cfg.deform_lr_factor                  # Multiplier for learning rate applied to the deformations
        self.repulse_extent = cfg.repulse_extent                    # Distance of repulsion for deformed kernel points

        #####################
        # Training parameters
        #####################

        # Maximal number of epochs
        self.max_epoch = cfg.max_epoch

        # Learning rate management
        learning_rate = 1e-2
        self.momentum = cfg.momentum
        lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, self.max_epoch)}
        self.grad_clip_norm = cfg.grad_clip_norm

        # Number of batch
        self.batch_num = cfg.batch_num

        # Number of steps per epochs
        self.epoch_steps = cfg.epoch_steps

        # Number of validation examples per epoch
        self.validation_size = cfg.validation_size

        # Number of epoch between each checkpoint
        self.checkpoint_gap = cfg.checkpoint_gap

        # Augmentations
        self.augment_scale_anisotropic = cfg.augment_scale_anisotropic
        self.augment_symmetries = cfg.augment_symmetries
        self.augment_rotation = cfg.augment_rotation
        self.augment_scale_min = cfg.augment_scale_min
        self.augment_scale_max = cfg.augment_scale_max
        self.augment_noise = cfg.augment_noise
        self.augment_color = cfg.augment_color

        # The way we balance segmentation loss
        #   > 'none': Each point in the whole batch has the same contribution.
        #   > 'class': Each class has the same contribution (points are weighted according to class balance)
        #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
        self.segloss_balance = cfg.segloss_balance

        # Do we nee to save convergence
        self.saving = cfg.saving
        self.saving_path = cfg.saving_path

        # Dataset folder
        self.path = cfg.path
        self.writer = SummaryWriter(cfg.writer)

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'LAS'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                            'resnetb',
                            'resnetb_strided',
                            'resnetb',
                            'resnetb',
                            'resnetb_strided',
                            'resnetb_deformable',
                            'resnetb_deformable',
                            'resnetb_deformable_strided',
                            'resnetb_deformable',
                            'resnetb_deformable',
                            'resnetb_deformable_strided',
                            'resnetb_deformable',
                            'resnetb_deformable',
                            'nearest_upsample',
                            'unary',
                            'nearest_upsample',
                            'unary',
                            'nearest_upsample',
                            'unary',
                            'nearest_upsample',
                            'unary']


if __name__ == "__main__":
    my_app()