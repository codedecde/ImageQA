#!/usr/bin/env python

import datetime
import os
import sys
import log
import logging
import argparse
import math

from optimization_weight import *
from co_wbw_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *
from configurations import get_configurations

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']

def train(options):

    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')

    data_provision_att_vqa = DataProvisionAttVqa(options['data_path'],
                                                 options['feature_file'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############
    
    if not options['load_model']:
        params = init_params(options)
        shared_params = init_shared_params(params)
    else:
        options, params, shared_params = load_model(options['model_file'])

    image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, alpha, pred, confidence = build_model(shared_params, options)
    logger.info('finished building model')

    ####################
    # add weight decay #
    ####################
    weight_decay = theano.shared(numpy.float32(options['weight_decay']),\
                                 name = 'weight_decay')
    reg_cost = 0

    for k in shared_params.iterkeys():
        if k != 'w_emb':
            reg_cost += (shared_params[k]**2).sum()

    reg_cost *= weight_decay
    reg_cost = cost + reg_cost

    ###############
    # # gradients #
    ###############
    grads = T.grad(reg_cost, wrt = shared_params.values())
    grad_buf = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k )
                for k, p in shared_params.iteritems()]
    # accumulate the gradients within one batch
    update_grad = [(g_b, g) for g_b, g in zip(grad_buf, grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']
    grad_norm = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf]
    update_clip = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b*grad_clip/g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm, grad_buf)]

    # corresponding update function
    f_grad_clip = theano.function(inputs = [],
                                  updates = update_clip)
    f_output_grad_norm = theano.function(inputs = [],
                                         outputs = grad_norm)
    f_train = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                              outputs = [cost, accu],
                              updates = update_grad,
                              on_unused_input='warn')
    # validation function no gradient updates
    f_val = theano.function(inputs = [image_feat, input_idx, input_mask, label],
                            outputs = [cost, accu],
                            on_unused_input='warn')
    # the attention function to get the attention and the predicted label
    f_att_pred = theano.function(inputs = [image_feat, input_idx, input_mask], 
                                 outputs = [alpha, pred],
                                 on_unused_input='warn')

    f_grad_cache_update, f_param_update \
        = eval(options['optimization'])(shared_params, grad_buf, options)
    logger.info('finished building function')

    # calculate how many iterations we need
    num_iters_one_epoch = data_provision_att_vqa.get_size(options['train_split']) / batch_size
    max_iters = max_epochs * num_iters_one_epoch
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']
    disp_interval = options['disp_interval']

    best_val_accu = 0.0
    best_param = dict()

    for itr in xrange(max_iters + 1):
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            val_cost_list = []
            val_accu_list = []
            my_count = 0
            val_count = 0
            dropout.set_value(numpy.float32(0.))
            for batch_image_feat, batch_question, batch_answer_label, batch_answer_counter \
                in data_provision_att_vqa.iterate_batch_with_counter(options['val_split'],
                                                    batch_size):
                input_idx, input_mask \
                    = process_batch(batch_question,
                                    reverse=options['reverse'])
                batch_image_feat = reshape_image_feat(batch_image_feat,
                                                      options['num_region'],
                                                      options['region_dim'])
                [cost, accu] = f_val(batch_image_feat, input_idx, input_mask,
                                     batch_answer_label.astype('int32').flatten())

                [_,pred] = f_att_pred(batch_image_feat, input_idx, input_mask)
            
                for i in xrange(len(pred)):
                    if batch_answer_counter[i].setdefault(pred[i],0) >= 3:
                        my_count += 1

                val_count += batch_image_feat.shape[0]
                val_cost_list.append(cost * batch_image_feat.shape[0])
                val_accu_list.append(accu * batch_image_feat.shape[0])
            ave_val_cost = sum(val_cost_list) / float(val_count)
            ave_val_accu = sum(val_accu_list) / float(val_count)
            if best_val_accu < ave_val_accu:
                best_val_accu = ave_val_accu
                shared_to_cpu(shared_params, best_param)
            logger.info('validation cost: %f accu: %f' %(ave_val_cost, ave_val_accu))
            logger.info('Paper ACCURACY : %f' %(float(my_count)/val_count))

        dropout.set_value(numpy.float32(1.))
        if options['sample_answer']:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch_sample(options['train_split'],
                                                       batch_size)
        else:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch(options['train_split'], batch_size)
        input_idx, input_mask \
            = process_batch(batch_question, reverse=options['reverse'])
        batch_image_feat = reshape_image_feat(batch_image_feat,
                                              options['num_region'],
                                              options['region_dim'])

        [cost, accu] = f_train(batch_image_feat, input_idx, input_mask,
                               batch_answer_label.astype('int32').flatten())
        # output_norm = f_output_grad_norm()
        # logger.info(output_norm)
        # pdb.set_trace()
        f_grad_clip()
        f_grad_cache_update()
        lr_t = get_lr(options, itr / float(num_iters_one_epoch))
        f_param_update(lr_t)

        if options['shuffle'] and itr > 0 and itr % num_iters_one_epoch == 0:
            data_provision_att_vqa.random_shuffle()

        if (itr % disp_interval) == 0  or (itr == max_iters):
            logger.info('iteration %d/%d epoch %f/%d cost %f accu %f, lr %f' \
                        % (itr, max_iters,
                           itr / float(num_iters_one_epoch), max_epochs,
                           cost, accu, lr_t))
            if np.isnan(cost):
                logger.info('nan detected')
                file_name = options['model_name'] + '_nan_debug.model'
                logger.info('saving the debug model to %s' %(file_name))
                save_model(os.path.join(options['expt_folder'], file_name), options,
                           best_param)
                return 0


    logger.info('best validation accu: %f', best_val_accu)
    file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_accu) + '.model'
    logger.info('saving the best model to %s' %(file_name))
    save_model(os.path.join(options['expt_folder'], file_name), options,
               best_param)

    return best_val_accu

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
    train(options)
