import os
import random

import pandas as pd
import numpy as np
import h5py

import torch
import os


import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def read_metadata(metadata_path):
    '''Read metadata from a csv file.

    Returns:
      meta_dict: dict of meta data, e.g.:
        {'audio_name': np.array(['a.wav', 'b.wav', ...]),
         'scene_label': np.array(['airport', 'bus', ...]),
         ...}
    '''
    df = pd.read_csv(metadata_path, sep='\t')

    meta_dict = {}

    meta_dict['audio_name'] = np.array(
        [name.split('/')[1] for name in df['filename'].tolist()])

    if 'scene_label' in df.keys():
        meta_dict['scene_label'] = np.array(df['scene_label'])

    if 'identifier' in df.keys():
        meta_dict['identifier'] = np.array(df['identifier'])

    if 'source_label' in df.keys():
        meta_dict['source_label'] = np.array(df['source_label'])

    return meta_dict


def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 2)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]

    scalar = {'mean': mean, 'std': std}
    return scalar


def scale(x, mean, std):
    return (x - mean) / std

def normalization(x):

    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range

def standardization(x):

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


def inverse_scale(x, mean, std):
    return x * std + mean

def sparse_to_categorical(x, n_out):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))

def mixup_data(x, y, alpha=0.2):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)

def trans(x):

    return x[:,None,:,:]

def pca_show(labels,index,output,target,source):


    target_name = []
    pca = TSNE(n_components=2, learning_rate=100, init='pca')
    prob1 = StandardScaler().fit_transform(output)
    principalComponents = pca.fit_transform(output)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    # print(pca.explained_variance_ratio_)
    for i in target:
        target_name.append(index[i])

    df = pd.DataFrame(target_name, columns=['target'])
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    plt.tick_params(labelsize=25)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA Results For {}'.format(str(source).upper()), fontsize=20)

    for label in labels:
        indicesToKeep = finalDf['target'] == label
        ax.scatter(principalComponents[indicesToKeep, 0]
                   , principalComponents[indicesToKeep, 1]
                   , s=100)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ax.legend(labels,loc='center', bbox_to_anchor=(0.5, 1.2), ncol=5)


    #ax.legend(labels)
    ax.grid()

    return plt.show()
def frequency_masking(mel_spectrogram, frequency_masking_para=40, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0 + f), :] = 0
    return mel_spectrogram


def time_masking(mel_spectrogram, time_masking_para=10, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0 + t)] = 0
    return mel_spectrogram
