import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
import pandas as pd
import h5py
from utilities.utlies import scale, read_metadata,load_scalar,sparse_to_categorical,sparse_to_categorical,normalization,standardization
from utilities import config
import logging
import time
import os
import random
from tecent.funcs import *
from tecent.utils import *
class Base(object):

    def __init__(self):
        '''Base class for data generator
        '''
        pass

    def load_hdf5(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature'] = hf['feature'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

class DataGenerator(Base):

    def __init__(self, feature_hdf5_path, train_csv, evaluate_csv,
                 scalar, batch_size, seed=1234,crop_length=600,alpha=0.5):
        '''Data generator for training and validation.

        '''
        self.alpha = alpha
        self.scalar = scalar
        self.batch_size = batch_size
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.state = np.random.get_state()
        self.all_classes_num = len(config.labels)


        self.lb_to_idx = config.lb_to_idx
        self.idx_to_lb = config.idx_to_lb
        self.NewLength = crop_length
        # Load training data
        load_time = time.time()

        self.data_dict = self.load_hdf5(feature_hdf5_path)

        train_meta = read_metadata(train_csv)
        validate_meta = read_metadata(evaluate_csv)

        self.source_audio_indexes ,self.target_audio_indexes,self.only_source_audio_indexes,self.train_audio_indexes= self.get_audio_indexes(
            train_meta, self.data_dict,  'train')

        self.validate_audio_indexes,self.pair_source,self.pair_target = self.get_audio_indexes(
            validate_meta, self.data_dict,  'validate')



        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))

        logging.info('Training audio num: {}'.format(len(self.source_audio_indexes)+len(self.target_audio_indexes)+len(self.only_source_audio_indexes)))

        logging.info('Source or Targrt audio num: {}'.format(len(self.source_audio_indexes)))

        logging.info('Only Source audio num: {}'.format(len(self.only_source_audio_indexes)))

        logging.info('Validation audio num: {}'.format(len(self.validate_audio_indexes)))

        self.pointer = 0
        self.pointer2 = 0
        random.seed(self.seed)
        random.shuffle(self.source_audio_indexes)
        random.seed(self.seed)
        random.shuffle(self.target_audio_indexes)
        random.seed(self.seed)
        random.shuffle(self.only_source_audio_indexes)
        random.seed(self.seed)
        random.shuffle(self.train_audio_indexes)

    def get_audio_indexes(self, meta_data, data_dict, data_type):
        '''Get train or validate indexes.
        '''
        train_audio_index=[]

        if data_type == 'train':
            source_indexes = []
            target_indexes = []
            only_source_indexes = []

            for name in meta_data['audio_name']:
                if 'a.wav' in name:
                    loct = np.argwhere(data_dict['audio_name'] == name)

                    if len(loct) > 0:
                        index = loct[0, 0]

                        only_source_indexes.append(index)
                        train_audio_index.append(index)

                elif 'b.wav' or 'c.wav' in name:

                    loct = np.argwhere(data_dict['audio_name'] == name)
                    en_loct = np.argwhere(data_dict['audio_name'] == (name[0:-5]+'a.wav'))

                    if len(loct) and len(en_loct) > 0:
                        index = loct[0, 0]
                        en_index = en_loct[0,0]
                        source_indexes.append(en_index)
                        target_indexes.append(index)
                        train_audio_index.append(index)

                if 's1.wav' or 's2.wav' or 's3.wav' in name:


                    loct = np.argwhere(data_dict['audio_name'] == name)
                    en_loct = np.argwhere(data_dict['audio_name'] == (name[0:-6] + 'a.wav'))

                    if len(loct) and len(en_loct)> 0:
                        index = loct[0, 0]
                        en_index = en_loct[0, 0]
                        source_indexes.append(en_index)
                        target_indexes.append(index)
                        train_audio_index.append(index)

            return np.array(source_indexes),np.array(target_indexes),np.array(only_source_indexes),np.array(train_audio_index)

        if data_type=='validate':
            audio_indexes = []
            pair_source = []
            pair_target = []

            for name in meta_data['audio_name']:
                if 'a.wav' in name:
                    loct = np.argwhere(data_dict['audio_name'] == name)

                    if len(loct) > 0:
                        index = loct[0, 0]

                        audio_indexes.append(index)

                elif 'b.wav' or 'c.wav' in name and name[0:-5]+'a.wav' in meta_data['audio_name']:

                    loct = np.argwhere(data_dict['audio_name'] == name)
                    en_loct = np.argwhere(data_dict['audio_name'] == (name[0:-5]+'a.wav'))

                    if len(loct) and len(en_loct) > 0:
                        index = loct[0, 0]
                        en_index = en_loct[0,0]
                        pair_target.append(index)
                        pair_source.append(en_index)

                elif 's1.wav' or 's2.wav' or 's3.wav' in name and name[0:-6]+'a.wav' in meta_data['audio_name']:


                    loct = np.argwhere(data_dict['audio_name'] == name)
                    en_loct = np.argwhere(data_dict['audio_name'] == (name[0:-6] + 'a.wav'))

                    if len(loct) and len(en_loct)> 0:
                        index = loct[0, 0]
                        en_index = en_loct[0, 0]
                        pair_target.append(index)
                        pair_source.append(en_index)

            for name in meta_data['audio_name']:
                if 'a.wav' not in name:
                    loct = np.argwhere(data_dict['audio_name'] == name)

                    if len(loct) > 0:
                        index = loct[0, 0]

                        audio_indexes.append(index)


            return np.array(audio_indexes),np.array(pair_source),np.array(pair_target)

    def generate_train_domain(self):
        '''Generate mini-batch data for training.

        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''

        while True:
            if self.pointer+self.batch_size*2 >= len(self.target_audio_indexes):
                self.pointer = 0

            random.seed(self.seed)
            random.shuffle(self.source_audio_indexes)
            random.seed(self.seed)
            random.shuffle(self.target_audio_indexes)


            if self.pointer2+self.batch_size*2 >=len(self.only_source_audio_indexes):
                self.pointer2=0
                random.seed(self.seed)
                random.shuffle(self.only_source_audio_indexes)

        # Get batch audio_indexes
            batch_source_audio_indexes, \
            batch_target_audio_indexes,\
            batch_only_source_indexes = \
            self.source_audio_indexes[self.pointer: self.pointer + self.batch_size*2],\
            self.target_audio_indexes[self.pointer: self.pointer + self.batch_size*2],\
            self.only_source_audio_indexes[self.pointer2: self.pointer2 + self.batch_size*2]

            self.pointer += self.batch_size
            self.pointer2 += self.batch_size

            batch_data_dict = {}

            #batch_source_feature = self.data_dict['feature'][batch_source_audio_indexes]
            #batch_target_feature = self.data_dict['feature'][batch_target_audio_indexes]

            #batch_only_source_feature = self.data_dict['feature'][batch_only_source_indexes]
            #batch_source_feature, sparse_source_target = self.__data_generation_mixup(batch_source_audio_indexes,
            #                                                                             self.data_dict['feature'],
            #                                                                            self.data_dict['target'])

            batch_source_feature,batch_target_feature, sparse_target_target = self.__data_generation_domain_mixup(batch_source_audio_indexes,
                                                                                          batch_target_audio_indexes,
                                                                                          self.data_dict['feature'],
                                                                                          self.data_dict['target'])

            batch_only_source_feature,sparse_only_source_target =self.__data_generation_mixup(batch_only_source_indexes,
                                                                                        self.data_dict['feature'],
                                                                                        self.data_dict['target'])

            batch_data_dict['source_feature'] = batch_source_feature
            batch_data_dict['target_feature'] = batch_target_feature
            batch_data_dict['only_source_feature'] = batch_only_source_feature

            #sparse_target = self.data_dict['target'][batch_source_audio_indexes]

            #sparse_only_source_target = self.data_dict['target'][batch_only_source_indexes]

            #batch_data_dict['target'] = sparse_to_categorical(sparse_target, self.all_classes_num)
            batch_data_dict['target'] = sparse_target_target
            batch_data_dict['only_source_target'] = sparse_only_source_target

            #sparse_to_categorical(sparse_only_source_target, self.all_classes_num)

            yield batch_data_dict

    def __data_generation_mixup(self, batch_ids, X_train, y_train):
        _, t,f  = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X_train = self.transform(X_train[batch_ids])

        X1 = X_train[:self.batch_size]
        X2 = X_train[self.batch_size:]

        y_train = sparse_to_categorical(y_train[batch_ids],self.all_classes_num)
        # for j in range(X1.shape[0]):
        #     # spectrum augment
        #
        #     X1[j, :, :] = frequency_masking(X1[j, :, :])
        #     X1[j, :, :] = time_masking(X1[j, :, :])
        #     X2[j, :, :] = frequency_masking(X2[j, :, :])
        #     X2[j, :, :] = time_masking(X2[j, :, :])
        #
        #     # random cropping
        #     StartLoc1 = np.random.randint(0, X1.shape[1] - self.NewLength)
        #     StartLoc2 = np.random.randint(0, X2.shape[1] - self.NewLength)
        #
        #     X1[j, 0:self.NewLength, :] = X1[j, StartLoc1:StartLoc1 + self.NewLength, :]
        #     X2[j, 0:self.NewLength, :] = X2[j, StartLoc2:StartLoc2 + self.NewLength, :]
        #
        # X1 = X1[:, 0:self.NewLength, :]
        # X2 = X2[:, 0:self.NewLength, :]
        #
        # mixup
        X = X1 * X_l + X2 * (1.0 - X_l)


        y1 = y_train[:self.batch_size]
        y2 = y_train[self.batch_size:]
        y = y1 * y_l + y2 * (1.0 - y_l)

        return X, y

    def __data_generation_domain_mixup(self, batch_ids_source,batch_ids_target, X_train, y_train):
        _, t,f  = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X_train_source = self.transform(X_train[batch_ids_source])
        X_train_target = self.transform(X_train[batch_ids_target])

        X1_source = X_train_source[:self.batch_size]
        X2_source = X_train_source[self.batch_size:]

        X1_target = X_train_target[:self.batch_size]
        X2_target = X_train_target[self.batch_size:]

        y_train = sparse_to_categorical(y_train[batch_ids_source],self.all_classes_num)
        for j in range(X1_source.shape[0]):
            # spectrum augment

            X1_source[j, :, :] = frequency_masking(X1_source[j, :, :])
            X1_source[j, :, :] = time_masking(X1_source[j, :, :])
            X2_source[j, :, :] = frequency_masking(X2_source[j, :, :])
            X2_source[j, :, :] = time_masking(X2_source[j, :, :])

            X1_target[j, :, :] = frequency_masking(X1_target[j, :, :])
            X1_target[j, :, :] = time_masking(X1_target[j, :, :])
            X2_target[j, :, :] = frequency_masking(X2_target[j, :, :])
            X2_target[j, :, :] = time_masking(X2_target[j, :, :])

            # random cropping
            StartLoc1 = np.random.randint(0, X1_source.shape[1] - self.NewLength)
            StartLoc2 = np.random.randint(0, X2_source.shape[1] - self.NewLength)

            X1_source[j, 0:self.NewLength, :] = X1_source[j, StartLoc1:StartLoc1 + self.NewLength, :]
            X2_source[j, 0:self.NewLength, :] = X2_source[j, StartLoc2:StartLoc2 + self.NewLength, :]
            X1_target[j, 0:self.NewLength, :] = X1_target[j, StartLoc1:StartLoc1 + self.NewLength, :]
            X2_target[j, 0:self.NewLength, :] = X2_target[j, StartLoc2:StartLoc2 + self.NewLength, :]


        X1_source = X1_source[:, 0:self.NewLength, :]
        X2_source = X2_source[:, 0:self.NewLength, :]
        X1_target = X1_target[:, 0:self.NewLength, :]
        X2_target = X2_target[:, 0:self.NewLength, :]


        # mixup
        X_source = X1_source * X_l + X2_source * (1.0 - X_l)
        X_target = X1_target * X_l + X2_target * (1.0 - X_l)


        y1 = y_train[:self.batch_size]
        y2 = y_train[self.batch_size:]
        y = y1 * y_l + y2 * (1.0 - y_l)

        return X_source, X_target,y

    def __data_generation_domain(self, batch_ids, X_train, y_train):
        _, t,f  = X_train.shape


        X = X_train[batch_ids]
        X = self.transform(X)

        y_train = sparse_to_categorical(y_train,self.all_classes_num)
        for j in range(X.shape[0]):
            # spectrum augment

            X[j,  :, :] = frequency_masking(X[j, :, :])
            X[j,  :, :] = time_masking(X[j, :, :])

            # random cropping
            StartLoc1 = np.random.randint(0, X.shape[1] - self.NewLength)


            X[j,  0:self.NewLength, :] = X[j,  StartLoc1:StartLoc1 + self.NewLength, :]


        X = X[:,  0:self.NewLength, :]

        y = y_train[batch_ids]




        return X, y

    def __data_generation(self, batch_ids, X_train, y_train):
        _, t,f  = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X_train = self.transform(X_train)

        X1 = X_train[batch_ids[:self.batch_size]]
        X2 = X_train[batch_ids[self.batch_size:]]

        y_train = sparse_to_categorical(y_train,self.all_classes_num)
        for j in range(X1.shape[0]):
            # spectrum augment

            X1[j,  :, :] = frequency_masking(X1[j, :, :])
            X1[j,  :, :] = time_masking(X1[j, :, :])
            X2[j,  :, :] = frequency_masking(X2[j, :, :])
            X2[j,  :, :] = time_masking(X2[j, :, :])

            # random cropping
            StartLoc1 = np.random.randint(0, X1.shape[1] - self.NewLength)
            StartLoc2 = np.random.randint(0, X2.shape[1] - self.NewLength)

            X1[j,  0:self.NewLength, :] = X1[j,  StartLoc1:StartLoc1 + self.NewLength, :]
            X2[j,  0:self.NewLength, :] = X2[j,  StartLoc2:StartLoc2 + self.NewLength, :]

        X1 = X1[:,  0:self.NewLength, :]
        X2 = X2[:,  0:self.NewLength, :]

        # mixup
        X = X1 * X_l + X2 * (1.0 - X_l)

        if isinstance(y_train, list):
            y = []

            for y_train_ in y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = y_train[batch_ids[:self.batch_size]]
            y2 = y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        return X, y

    def generate_train(self):
        '''Generate mini-batch data for training.

        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''

        while True:
            if self.pointer >= len(self.only_source_audio_indexes):
                self.pointer = 0
                random.seed(self.seed)
                random.shuffle(self.only_source_audio_indexes)

        # Get batch audio_indexes
            batch_audio_indexes = self.only_source_audio_indexes[self.pointer: self.pointer + self.batch_size]

            self.pointer += self.batch_size

            batch_data_dict = {}
            '''
            batch_data_dict['audio_name'] = \
                self.data_dict['audio_name'][batch_audio_indexes]
            batch_data_dict['encoder_audio_name'] = \
                self.data_dict['audio_name'][batch_encoder_indexes]
            '''
            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)

            batch_data_dict['feature'] = batch_feature

            #batch_data_dict['target'] = self.data_dict['target'][batch_audio_indexes]
            sparse_target = self.data_dict['target'][batch_audio_indexes]
            batch_data_dict['target'] = sparse_to_categorical(
                sparse_target, self.all_classes_num)
            yield batch_data_dict

    def generate_all_train(self):
        '''Generate mini-batch data for training.

        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''

        while True:
            if self.pointer >= len(self.train_audio_indexes):
                self.pointer = 0
                random.seed(self.seed)
                random.shuffle(self.train_audio_indexes)

        # Get batch audio_indexes
            batch_audio_indexes = self.train_audio_indexes[self.pointer: self.pointer + self.batch_size]

            self.pointer += self.batch_size

            batch_data_dict = {}

            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)

            batch_data_dict['feature'] = batch_feature

            sparse_target = self.data_dict['target'][batch_audio_indexes]
            batch_data_dict['target'] = sparse_to_categorical(
                sparse_target, self.all_classes_num)

            yield batch_data_dict

    def get_source_indexes(self, indexes, data_dict, source):
        '''Get indexes of specific source.
        '''
        source_indexes = np.array([index for index in indexes if data_dict['source_label'][index] == source])

        return source_indexes

    def generate_evaluate(self, data_type, source, max_iteration=None):
        '''Generate mini-batch data for training.

        Args:
          data_type: 'train' | 'validate'
          source: 'a' | 'b' | 'c'
          max_iteration: int, maximum iteration to validate to speed up validation

        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)
        elif data_type == 'evaluate':
            audio_indexes = np.array(self.validate_audio_indexes)
        else:
            raise Exception('Incorrect argument!')

        audio_indexes = self.get_source_indexes(
            audio_indexes, self.data_dict, source)

        iteration = 0
        pointer = 0
        logging.info('evaluate_num:{}'.format(len(audio_indexes)))
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                self.data_dict['audio_name'][batch_audio_indexes]

            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature

            batch_data_dict['target']  = self.data_dict['target'][batch_audio_indexes]

            yield batch_data_dict

    def generate_evaluate_pair(self, source, max_iteration=None):

        global audio_indexes
        batch_size = self.batch_size

        pair_source = np.array(self.pair_source)
        pair_target = np.array(self.pair_target)
        if source == 'source':
            audio_indexes = pair_source
        elif source == 'target':
            audio_indexes = pair_target

        iteration = 0
        pointer = 0
        logging.info('evaluate_num:{}'.format(len(audio_indexes)))
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                self.data_dict['audio_name'][batch_audio_indexes]

            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature

            batch_data_dict['target'] = self.data_dict['target'][batch_audio_indexes]

            yield batch_data_dict


    def transform(self,x):

        return scale(x, self.scalar['mean'], self.scalar['std'])


