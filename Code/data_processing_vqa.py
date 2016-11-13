#!/usr/bin/env python

import numpy as np

def process_batch(batch_question, reverse=False):
    question_length = []
    for question in batch_question:
        question_length.append(len(question))
    max_length = max(question_length)
    input_idx = np.zeros((max_length, batch_question.shape[0]), dtype='int32')
    input_mask = np.zeros((max_length, batch_question.shape[0]), dtype='float32')
    for i, question in enumerate(batch_question):
        if reverse:
            input_idx[0:question_length[i], i] = question[0:question_length[i]][::-1]
        else:
            input_idx[0:question_length[i], i] = question[0:question_length[i]]
        input_mask[0:question_length[i], i] = 1.0
    return input_idx, input_mask

def get_label_idx(n_output, batch_size):
    label_idx = np.zeros((n_output, batch_size), dtype='int32')
    for i in range(n_output):
        label_idx[i] = i
    return label_idx

def reshape_image_feat(batch_image_feat, num_region, region_dim):
    return np.reshape(np.asarray(batch_image_feat), (batch_image_feat.shape[0],
                                                     num_region, region_dim))
