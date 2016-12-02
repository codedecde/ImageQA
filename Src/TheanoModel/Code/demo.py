#!/usr/bin/env python

import datetime
import os
import sys
import log
import logging
import argparse
import math
import pickle
from optimization_weight import *
from wbw_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *
from configurations import get_configurations

def AnswerQuestion(image_id, question, id):
    global cid_mid, question_dict, data, answer_dict
    old_question = question
    old_id = image_id
    image_id = cid_mid[image_id]
    image_feat = data._image_feat[image_id]
    image_feat = reshape_image_feat(image_feat.todense(), options['num_region'], options['region_dim'])

    question = [question_dict[word] for word in question.lower().split(' ')]
    batch_question = np.array([question])
    input_idx, input_mask = process_batch(batch_question, reverse=False)   

    [alpha, pred, confidence] = f_att_pred(image_feat, input_idx, input_mask)
    pickle.dump([alpha, pred,confidence , old_question.split(' '), old_id], open(options['att_expt_folder']+'alphavec'+str(id)+'.npy','wb'))
    logger.info("\n\nAnswer      : %s\nConfidence  : %f\n"%(inv_ans_dict[pred[0]], confidence))
    return inv_ans_dict[pred[0]], confidence


if __name__ == '__main__':
    theano.config.optimizer='fast_compile' 
    logger = log.setup_custom_logger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('changes', nargs='*',
                        help='Changes to default values',
                        default = '')
    args = parser.parse_args()
    for change in args.changes:
        logger.info('dict({%s})'%(change))
        options.update(eval('dict({%s})'%(change)))
    options = get_configurations()

    ## Load Standard Dicts
    image_list = pickle.load(open(options['data_path']+'image_list.pkl'))
    cid_mid = {image_list[x]:x for x in range(len(image_list))}
    question_dict = pickle.load(open(options['data_path']+'question_dict.pkl'))
    inv_ques_dict = {question_dict[x]:x for x in question_dict}
    answer_dict = pickle.load(open(options['data_path']+'answer_dict.pkl'))
    inv_ans_dict = {answer_dict[x]:x for x in answer_dict}

    ## Load Data
    data = DataProvisionAttVqa(options['data_path'], options['feature_file'])

    ## Load Model
    options, params, shared_params = load_model(options['model_file'])
    logger.info('finished loading Model')
    image_feat, input_idx, input_mask, label, dropout, cost, accu, alpha, pred, confidence = build_model(shared_params, options)
    logger.info('finished Building Model')
    f_att_pred = theano.function(inputs = [image_feat, input_idx, input_mask], 
                                 outputs = [alpha, pred, confidence],
                                 on_unused_input='warn')
    
    options['att_expt_folder'] = '/home/ee/btech/ee1130798/DL/Proj/AttentionVectors/'
    logger.info('Starting Questions ')
    	
    AnswerQuestion(176265, 'Is the color of the cake the same as the color of the balloons', 0)
    AnswerQuestion(176265, 'How many people are here', 1)
    AnswerQuestion(176265, 'Is that a type of food normally seen at weddings', 2)
    AnswerQuestion(176265, 'Are the two people in white coats surprised', 3)
    AnswerQuestion(176265, 'What color are the balloons', 4)
    AnswerQuestion(176265, 'Are there any balloons', 5)

