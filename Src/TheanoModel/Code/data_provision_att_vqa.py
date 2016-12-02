#!/usr/bin/env python

import pdb
import random
import numpy as np
import pickle as pkl
import os
import log
import logging
import h5py
import scipy.sparse
import scipy.sparse as sparse
from collections import OrderedDict

logger = logging.getLogger('root')

class DataProvisionAttVqa:
    def __init__(self, data_folder, feature_file):
        self._image_feat = self.load_image_feat(data_folder, feature_file)
        self._question_id = OrderedDict()
        self._image_id = OrderedDict()
        self._question = OrderedDict()
        # answer set
        self._answer = OrderedDict()
        # answer counter
        self._answer_counter = OrderedDict()
        # most common answer
        self._answer_label = OrderedDict()
        self._splits = ['train', 'val1', 'val2', 'val2_all']
        self._pointer = OrderedDict()
        for split in self._splits:
            with open(os.path.join(data_folder, split) + '.pkl') as f:
                split_question_id = pkl.load(f)
                split_image_id = pkl.load(f)
                split_question = pkl.load(f)
                split_answer = pkl.load(f)
                split_answer_counter = pkl.load(f)
                split_answer_label = pkl.load(f)
            idx = range(split_question.shape[0])
            if not (split is 'val2' or split is 'val2_all'):
                random.shuffle(idx)
            self._question_id[split] = split_question_id[idx]
            self._image_id[split] = split_image_id[idx]
            self._question[split] = split_question[idx]
            self._answer[split] = split_answer[idx]
            self._answer_counter[split] = split_answer_counter[idx]
            self._answer_label[split] = split_answer_label[idx]
            self._pointer[split] = 0
        self._splits.append('trainval1')
        self._question_id['trainval1'] = np.concatenate([self._question_id['train'],
                                                         self._question_id['val1']],
                                                        axis = 0)
        self._image_id['trainval1'] = np.concatenate([self._image_id['train'],
                                                      self._image_id['val1']],
                                                     axis = 0)
        self._question['trainval1'] = np.concatenate([self._question['train'],
                                                      self._question['val1']],
                                                     axis = 0)
        self._answer['trainval1'] = np.concatenate([self._answer['train'],
                                                    self._answer['val1']],
                                                   axis = 0)
        self._answer_counter['trainval1'] \
            = np.concatenate([self._answer_counter['train'],
                              self._answer_counter['val1']],
                             axis = 0)
        self._answer_label['trainval1'] \
            = np.concatenate([self._answer_label['train'],
                              self._answer_label['val1']],
                             axis = 0)
        self._pointer['trainval1'] = 0

        logger.info('finished loading data')

    def load_image_feat(self, data_path, h5_file):
        image_h5 = h5py.File(os.path.join(data_path, h5_file), 'r')
        shape = image_h5['shape']
        data = image_h5['data']
        col_idx = image_h5['indices']
        count_idx = image_h5['indptr']
        return scipy.sparse.csr_matrix((data, col_idx, count_idx),
                                       dtype='float32',
                                       shape=(shape[0], shape[1]))

    def random_shuffle(self):
        for split in self._splits:
            if not (split is 'val2' or split is 'val2_all'):
                idx = range(len(self._question[split]))
                random.shuffle(idx)
                self._question_id[split] = self._question_id[split][idx]
                self._image_id[split] = self._image_id[split][idx]
                self._question[split] = self._question[split][idx]
                self._answer[split] = self._answer[split][idx]
                self._answer_counter[split] = self._answer_counter[split][idx]
                self._answer_label[split] = self._answer_label[split][idx]
                self._pointer[split] = 0

    def get_size(self, partition):
        return self._question[partition].shape[0]

    def reset_pointer(self, partition):
        self._pointer[partition] = 0

    def iterate_batch(self, partition, batch_size):
        logger.debug('begin to iterate batch for %s'%(partition))
        current = 0
        while current + batch_size <= self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :
                                                       current + batch_size]
            batch_question = self._question[partition][current :
                                                       current + batch_size]
            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][current :
                                                               current + batch_size]
            yield batch_image_feat, batch_question, batch_answer_label
            current = current + batch_size
            logger.debug('iterating batch at current: %d'%(current))
        if current != self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :]
            batch_question = self._question[partition][current :]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer_label = self._answer_label[partition][current :]
            logger.debug('finished iterating batch for %s'%(partition))
            yield batch_image_feat, batch_question, batch_answer_label

    def iterate_batch_with_counter(self, partition, batch_size):
        logger.debug('begin to iterate batch for %s'%(partition))
        current = 0
        while current + batch_size <= self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :
                                                       current + batch_size]
            batch_question = self._question[partition][current :
                                                       current + batch_size]
            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][current :
                                                               current + batch_size]
            batch_answer_counter = self._answer_counter[partition][current :
                                                                   current + batch_size]
            yield batch_image_feat, batch_question, batch_answer_label, \
                batch_answer_counter
            current = current + batch_size
            logger.debug('iterating batch at current: %d'%(current))
        if current != self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :]
            batch_question = self._question[partition][current :]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer_label = self._answer_label[partition][current :]
            batch_answer_counter = self._answer_counter[partition][current :]
            logger.debug('finished iterating batch for %s'%(partition))
            yield batch_image_feat, batch_question, batch_answer_label,\
                batch_answer_counter

    def iterate_batch_with_counter_imageid(self, partition, batch_size):
        logger.debug('begin to iterate batch for %s'%(partition))
        current = 0
        while current + batch_size <= self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :
                                                       current + batch_size]
            batch_question = self._question[partition][current :
                                                       current + batch_size]
            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][current :
                                                               current + batch_size]
            batch_answer_counter = self._answer_counter[partition][current :
                                                                   current + batch_size]
            yield batch_image_id, batch_image_feat, batch_question, batch_answer_label, \
                batch_answer_counter
            current = current + batch_size
            logger.debug('iterating batch at current: %d'%(current))
        if current != self._question[partition].shape[0]:
            batch_image_id = self._image_id[partition][current :]
            batch_question = self._question[partition][current :]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer_label = self._answer_label[partition][current :]
            batch_answer_counter = self._answer_counter[partition][current :]
            logger.debug('finished iterating batch for %s'%(partition))
            yield batch_image_id, batch_image_feat, batch_question, batch_answer_label,\
                batch_answer_counter


    def next_batch(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= self._question[partition].shape[0]:
            batch_question = self._question[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_id = self._image_id[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][self._pointer[partition] :
                                                               self._pointer[partition]
                                                               + batch_size]
            # update pointer
            self._pointer[partition] = (self._pointer[partition] + batch_size) \
                                       % self._question[partition].shape[0]
            logger.debug('next batch at pointer: %d'%(self._pointer[partition]))
            return batch_image_feat, batch_question, batch_answer_label
        else:
            logger.debug('new epoch of data iteration')
            next_pointer = (self._pointer[partition] + batch_size) \
                           % self._question[partition].shape[0]
            batch_question = self._question[partition][self._pointer[partition]:]
            batch_question = np.append(batch_question,
                                       self._question[partition][:next_pointer],
                                       axis = 0)
            batch_image_id = self._image_id[partition][self._pointer[partition]:]
            batch_image_id = np.append(batch_image_id,
                                       self._image_id[partition][:next_pointer],
                                       axis = 0)

            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer_label = self._answer_label[partition][self._pointer[partition]:]
            batch_answer_label = np.append(batch_answer_label,
                                           self._answer_label[partition][:next_pointer],
                                           axis = 0)
            self._pointer[partition] = next_pointer
            logger.debug('next batch at pointer: %d'%(next_pointer))
            return batch_image_feat, batch_question, batch_answer_label

    def next_batch_sample(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= self._question[partition].shape[0]:
            batch_question = self._question[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_id = self._image_id[partition][self._pointer[partition] :
                                                       self._pointer[partition]
                                                       + batch_size]
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()
            batch_answer = self._answer[partition][self._pointer[partition]:
                                                   self._pointer[partition]
                                                   + batch_size]
            batch_answer_label = [random.choice(ans)for ans in batch_answer]
            batch_answer_label = np.array(batch_answer_label)
            # update pointer
            self._pointer[partition] = (self._pointer[partition] + batch_size) \
                                       % self._question[partition].shape[0]
            logger.debug('next batch at pointer: %d'%(self._pointer[partition]))
            return batch_image_feat, batch_question, batch_answer_label
        else:
            logger.debug('new epoch of data iteration')
            next_pointer = (self._pointer[partition] + batch_size) \
                           % self._question[partition].shape[0]
            batch_question = self._question[partition][self._pointer[partition]:]
            batch_question = np.append(batch_question,
                                       self._question[partition][:next_pointer],
                                       axis = 0)
            batch_image_id = self._image_id[partition][self._pointer[partition]:]
            batch_image_id = np.append(batch_image_id,
                                       self._image_id[partition][:next_pointer],
                                       axis = 0)

            # index - 1 as query for image feature
            batch_image_feat = self._image_feat[batch_image_id]
            batch_image_feat = batch_image_feat.todense()

            batch_answer = self._answer[partition][self._pointer[partition]:]
            batch_answer = np.append(batch_answer,
                                     self._answer[partition][:next_pointer],
                                     axis = 0)
            batch_answer_label = [random.choice(ans)for ans in batch_answer]
            batch_answer_label = np.array(batch_answer_label)

            self._pointer[partition] = next_pointer
            logger.debug('next batch at pointer: %d'%(next_pointer))
            return batch_image_feat, batch_question, batch_answer_label
