import os

import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from pytorch.cpresnet import  *
from utilities.utlies import create_folder, load_scalar,create_logging
from utilities import config
from pytorch.dataset_mixup import *
from pytorch.model import *
from pytorch.evaluate import Evaluator, StatisticsContainer
from pytorch.pytorch_utils import move_data_to_gpu, forward
from tensorboardX import SummaryWriter
from pytorch.loss import nll_loss
from pytorch.pytorch_utils import random_feature
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary



def inference_validation():
    '''Inference and calculate metrics on validation data.

    Args:
      dataset_dir: string, directory of dataset
      subtask: 'a' | 'b' | 'c', corresponds to 3 subtasks in DCASE2019 Task1
      data_type: 'development'
      workspace: string, directory of workspace
      model_type: string, e.g. 'Cnn_9layers'
      iteration: int
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    # Arugments & parameters
    dataset_dir = config.dataset_dir

    model_type = config.model_type
    workspace = config.workspace
    iteration = config.evaluate_iteration
    evaluate_accs = config.evaluate_accs
    batch_size = config.batch_size
    cuda = config.cuda and torch.cuda.is_available()

    filename = 'log'

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second

    source = config.sources_to_evaluate
    in_domain_classes_num = len(config.labels) - 1

    sub_dir = 'TAU-urban-acoustic-scenes-2020-mobile-development'
    prefix = ''


    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                             'fold1_train.csv')

    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                                'fold1_evaluate.csv')

    feature_hdf5_path = os.path.join(workspace, 'features',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))

    scalar_path = os.path.join(workspace, 'scalars',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))

    checkpoint_path = os.path.join(workspace, 'checkpoints', filename,
                                   '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                   '{}'.format(sub_dir),
                                   model_type, '{}_iterations_best{}.pth'.format(iteration,evaluate_accs))

    logs_dir = os.path.join(workspace, 'logs',
        'logmel_{}frames_{}melbins'.format( frames_per_second, mel_bins),
        '{}'.format(sub_dir), model_type)


    create_logging(logs_dir, 'w')


    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)

    model = Model().cuda()

    loss_func = nll_loss


    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])

    if cuda:
        model.cuda()
    summary(model, ( 640, 64))
    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path,
        train_csv=train_csv,
        evaluate_csv=validate_csv,
        scalar=scalar,
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model1=model,
        data_generator=data_generator,
        cuda=cuda)

    for source in config.sources_to_evaluate:
        evaluator.evaluate(data_type='evaluate', source=source, verbose=True, pca=True,pair=False)
        #evaluator.evaluate(data_type='evaluate', source='s6', verbose=False, pair=True, pca=True)


    #evaluator.evaluate(data_type='evaluate', source='s6', verbose=False,pair=True,pca = True)

    # Visualize log mel spectrogram

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    inference_validation()

