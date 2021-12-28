import os
import sys

#sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import datetime
import _pickle as cPickle
import sed_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utilities import  utlies
from pytorch.pytorch_utils import forward
from utilities import config
from utilities.utlies import pca_show
from mpl_toolkits.mplot3d import Axes3D

class Evaluator(object):
    def __init__(self, model,data_generator, cuda=True):
        '''Evaluator to evaluate prediction performance.

        Args:
          model: object
          data_generator: object
          subtask: 'a' | 'b' | 'c'
          cuda: bool
        '''

        self.model = model


        self.data_generator = data_generator
        self.index = config.idx_to_lb
        self.labels = config.labels
        self.cuda = cuda

        self.frames_per_second = config.frames_per_second
        self.labels = config.labels

        self.all_classes_num = len(config.labels)
        self.idx_to_lb = config.idx_to_lb
        self.lb_to_idx = config.lb_to_idx

    def evaluate(self, data_type, source, max_iteration=None, verbose=False,pair= False,pca=False):
        '''Evaluate the performance.

        Args:
          data_type: 'train' | 'validate'
          source: 'a' | 'b' | 'c'
          max_iteration: None | int, maximum iteration to run to speed up evaluation
          verbose: bool
        '''
        self.source = source
        global generate_func1, generate_func2

        if pair == False:
            generate_func = self.data_generator.generate_evaluate(
                data_type=data_type,
                source=self.source,
                max_iteration=max_iteration)
            # Forward
            output_dict = forward(
                model=self.model,
                generate_func=generate_func,
                cuda=self.cuda,
                return_target=True)

            output1 = output_dict['output1']
            output2 = output_dict['output2']
            output3 = output_dict['output3']  # (audios_num, in_domain_classes_num)

            target = output_dict['target']  # (audios_num, in_domain_classes_num)

            prob1 = np.exp(output1)  # Subtask a, b use log softmax as output
            prob2 = np.exp(output2)
            prob3 = np.exp(output3)
            # Evaluate

            y_true = target

            y_pred1 = np.argmax(prob1, axis=-1)
            y_pred2 = np.argmax(prob2, axis=-1)
            y_pred3 = np.argmax(prob3, axis=-1)

            if pca == True:
                pca_show(self.labels,self.index,prob1,target,source=self.source)


            confusion_matrix1 = metrics.confusion_matrix(
                y_true, y_pred1, labels=np.arange(self.all_classes_num))

            classwise_accuracy1 = np.diag(confusion_matrix1) \
                                  / np.sum(confusion_matrix1, axis=-1)

            confusion_matrix2 = metrics.confusion_matrix(
                y_true, y_pred2, labels=np.arange(self.all_classes_num))

            classwise_accuracy2 = np.diag(confusion_matrix2) \
                                  / np.sum(confusion_matrix2, axis=-1)

            confusion_matrix3 = metrics.confusion_matrix(
                y_true, y_pred3, labels=np.arange(self.all_classes_num))

            classwise_accuracy3 = np.diag(confusion_matrix3) \
                                  / np.sum(confusion_matrix3, axis=-1)

            logging.info('Data type: {}'.format(data_type))
            logging.info('    Source: {}'.format(source))

            logging.info('    Average accuracy1: {:.3f}'.format(np.mean(classwise_accuracy1)))

            if verbose:
                classes_num = len(classwise_accuracy1)
                for n in range(classes_num):
                    logging.info('{:<20}{:.3f}'.format(self.labels[n],
                                                       classwise_accuracy1[n]))

                logging.info(confusion_matrix1)

            statistics = {
                'accuracy': classwise_accuracy1,
                'confusion_matrix': confusion_matrix1}

            return statistics, np.mean(classwise_accuracy1), np.mean(classwise_accuracy2), np.mean(
                classwise_accuracy3), -np.max(output1, axis=-1)

        if pair == True:
            generate_func1 = self.data_generator.generate_evaluate_pair(
                source='source',
                max_iteration=max_iteration)
            generate_func2 = self.data_generator.generate_evaluate_pair(
                source='target',
                max_iteration=max_iteration)

            # Forward
            output_dict1 = forward(
                model=self.model,
                generate_func=generate_func1,
                cuda=self.cuda,
                return_target=True)
            output_dict2 = forward(
                model=self.model,
                generate_func=generate_func2,
                cuda=self.cuda,
                return_target=True)

            output1 = output_dict1['output1']
            output2 = output_dict2['output1']

            target1 = output_dict1['target']  # (audios_num, in_domain_classes_num)
            target2 = output_dict2['target']
            prob1 = np.exp(output1)  # Subtask a, b use log softmax as output
            prob2 = np.exp(output2)

            y_true1 = target1
            y_true2 = target2

            y_pred1 = np.argmax(prob1, axis=-1)
            y_pred2 = np.argmax(prob2, axis=-1)

            if pca == True:
                pca_show(self.labels, self.index, prob1, target1,source=self.source)
                pca_show(self.labels, self.index, prob2, target2,source=self.source)


            confusion_matrix1 = metrics.confusion_matrix(
                y_true1, y_pred1, labels=np.arange(self.all_classes_num))
            confusion_matrix2 = metrics.confusion_matrix(
                y_true2, y_pred2, labels=np.arange(self.all_classes_num))

            classwise_accuracy1 = np.diag(confusion_matrix1) \
                                 / np.sum(confusion_matrix1, axis=-1)
            classwise_accuracy2 = np.diag(confusion_matrix2) \
                                 / np.sum(confusion_matrix2, axis=-1)

            logging.info('source')
            logging.info('    Average accuracy1: {:.3f}'.format(np.mean(classwise_accuracy1)))
            logging.info('target')
            logging.info('    Average accuracy1: {:.3f}'.format(np.mean(classwise_accuracy2)))


            if verbose:
                classes_num = len(classwise_accuracy1)
                for n in range(classes_num):
                    logging.info('{:<20}{:.3f}'.format(self.labels[n],
                                                       classwise_accuracy1[n]))

                logging.info(confusion_matrix1)
                for n in range(classes_num):
                    logging.info('{:<20}{:.3f}'.format(self.labels[n],
                                                       classwise_accuracy2[n]))

                logging.info(confusion_matrix2)

            statistics = {
                'accuracy1': classwise_accuracy1,
                'confusion_matrix1': confusion_matrix1,
                'accuracy2': classwise_accuracy2,
                'confusion_matrix2': confusion_matrix2
            }

            return statistics,np.mean(classwise_accuracy1),np.mean(classwise_accuracy2)

    def visualize(self, data_type, source, max_iteration=None):
        '''Visualize log mel spectrogram of different sound classes.

        Args:
          data_type: 'train' | 'validate'
          source: 'a' | 'b' | 'c'
          max_iteration: None | int, maximum iteration to run to speed up evaluation
        '''
        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        frames_num = config.frames_num
        labels = config.labels
        in_domain_classes_num = len(config.labels) - 1
        idx_to_lb = config.idx_to_lb

        generate_func = self.data_generator.generate_validate(
            data_type=data_type,
            source=source,
            max_iteration=max_iteration)

        # Forward
        output_dict = forward(
            model=self.model1,
            generate_func=generate_func,
            cuda=self.cuda,
            return_input=True,
            return_target=True)

        # Plot log mel spectrogram of different sound classes
        rows_num = 3
        cols_num = 4

        fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))

        for k in range(in_domain_classes_num):
            for n, audio_name in enumerate(output_dict['audio_name']):
                if output_dict['target'][n, k] == 1:
                    title = idx_to_lb[k]
                    row = k // cols_num
                    col = k % cols_num
                    axs[row, col].set_title(title, color='r')
                    logmel = inverse_scale(output_dict['feature'][n], self.data_generator.scalar['mean'],
                                           self.data_generator.scalar['std'])
                    axs[row, col].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
                    axs[row, col].set_xticks([0, frames_num])
                    axs[row, col].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                    axs[row, col].xaxis.set_ticks_position('bottom')
                    axs[row, col].set_ylabel('Mel bins')
                    axs[row, col].set_yticks([])
                    break

        for k in range(in_domain_classes_num, rows_num * cols_num):
            row = k // cols_num
            col = k % cols_num
            axs[row, col].set_visible(False)

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training.

        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Statistics of device 'a', 'b' and 'c'
        self.statistics_dict = {'a': [], 'b': [], 'c': [],'s1': [], 's2': [], 's3': [],'s4': [], 's5': [], 's6': []}

    def append_and_dump(self, iteration, source, statistics):
        '''Append statistics to container and dump the container.

        Args:
          iteration: int
          source: 'a' | 'b' | 'c', device
          statistics: dict of statistics
        '''
        statistics['iteration'] = iteration
        self.statistics_dict[source].append(statistics)

        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))

