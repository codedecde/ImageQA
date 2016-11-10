#!/usr/bin/env python

import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy
import numpy as np
from collections import OrderedDict
import cPickle as pickle

from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = config.floatX

def shared_to_cpu(shared_params, params):
    for k, v in shared_params.iteritems():
        params[k] = v.get_value()

def cpu_to_shared(params, shared_params):
    for k, v in params.iteritems():
        shared_params[k].set_value(v)

def save_model(filename, options, params, shared_params=None):
    if not shared_params == None:
        shared_to_cpu(shared_params, params);
    model = OrderedDict()
    model['options'] = options
    model['params'] = params
    pickle.dump(model, open(filename, 'w'))

def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    options = model['options']
    params = model['params']
    shared_params = init_shared_params(params)
    return options, params, shared_params
    # return options, params, shared_params


def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are
    orthogonal.
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')


def init_weight(n, d, options):
    ''' initialize weight matrix
    options['init_type'] determines
    gaussian or uniform initlizaiton
    '''
    if options['init_type'] == 'gaussian':
        return (numpy.random.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        return ((numpy.random.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)

def init_convweight(w_shape, options):
    ''' initialize weight matrix of convolutional layer
    '''
    if options['init_type'] == 'gaussian':
        return numpy.random.randn(*w_shape).astype(floatX) * options['std']
    elif options['init_type'] == 'uniform':
        return ((numpy.random.rand(*w_shape) * 2 - 1) * options['range']).astype(floatX)

layers = {'ff': ('init_fflayer', 'fflayer'),
          'lstm': ('init_lstm_layer', 'lstm_layer'),
          'lstm_append': (None, 'lstm_append_layer')}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# initialize the parmaters
def init_params(options):
    ''' Initialize all the parameters
    '''
    params = OrderedDict()
    n_words = options['n_words']
    n_emb = options['n_emb']
    n_dim = options['n_dim']
    n_attention = options['n_attention']
    n_image_feat = options['n_image_feat']
    n_common_feat = options['n_common_feat']
    n_output = options['n_output']

    # embedding weights
    # params['w_emb'] = init_weight(n_words, n_emb, options)
    ## use the same initialization as BOW
    params['w_emb'] = ((numpy.random.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)

    n_filter = 0
    if options['use_unigram_conv']:
        params = init_fflayer(params, n_emb, options['num_filter_unigram'],
                              options, prefix='conv_unigram')
        n_filter += options['num_filter_unigram']
    if options['use_bigram_conv']:
        params = init_fflayer(params, 2 * n_emb, options['num_filter_bigram'],
                              options, prefix='conv_bigram')
        n_filter += options['num_filter_bigram']
    if options['use_trigram_conv']:
        params = init_fflayer(params, 3 * n_emb, options['num_filter_trigram'],
                              options, prefix='conv_trigram')
        n_filter += options['num_filter_trigram']

    params = init_fflayer(params, n_image_feat, n_filter, options,
                          prefix='image_mlp')

    # attention model based parameters
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='image_att_mlp_1')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='sent_att_mlp_1')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_1')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='image_att_mlp_2')
    params = init_fflayer(params, n_filter, n_attention, options,
                          prefix='sent_att_mlp_2')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_2')

    for i in range(options['combined_num_mlp']):
        if i == 0 and options['combined_num_mlp'] == 1:
            params = init_fflayer(params, n_filter, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == 0 and options['combined_num_mlp'] != 1:
            params = init_fflayer(params, n_filter, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == options['combined_num_mlp'] - 1 :
            params = init_fflayer(params, n_common_feat, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        else:
            params = init_fflayer(params, n_common_feat, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))

    return params

def init_shared_params(params):
    ''' return a shared version of all parameters
    '''
    shared_params = OrderedDict()
    for k, p in params.iteritems():
        shared_params[k] = theano.shared(params[k], name = k)

    return shared_params


# activation function for ff layer
def tanh(x):
    return T.tanh(x)

def relu(x):
    return T.maximum(x, np.float32(0.))

def linear(x):
    return x

def init_fflayer(params, nin, nout, options, prefix='ff'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options)
    params[prefix + '_b'] = np.zeros(nout, dtype='float32')
    return params

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh'):
    ''' fflayer: multiply weight then add bias
    '''
    return eval(act_func)(T.dot(x, shared_params[prefix + '_w']) +
                          shared_params[prefix + '_b'])

def init_convlayer(params, w_shape, options, prefix='conv'):
    ''' init conv layer
    '''
    params[prefix + '_w'] = init_convweight(w_shape, options)
    params[prefix + '_b'] = np.zeros(w_shape[0]).astype(floatX)
    return params

def convlayer(shared_params, x, options, prefix='conv', act_func='tanh'):
    return eval(act_func)(conv.conv2d(x, shared_params[prefix + '_w']) +
                          shared_params[prefix + '_b'].dimshuffle('x', 0, 'x', 'x'))

def maxpool_layer(shared_params, x, maxpool_shape, options):
    return downsample.max_pool_2d(x, maxpool_shape, ignore_border=False)

def dropout_layer(x, dropout, trng, drop_ratio=0.5):
    ''' dropout layer
    '''
    x_drop = T.switch(dropout,
                      (x * trng.binomial(x.shape,
                                         p = 1 - drop_ratio,
                                         n = 1,
                                         dtype = x.dtype) \
                       / (numpy.float32(1.0) - drop_ratio)),
                      x)
    return x_drop

def similarity_layer(feat, feat_seq):
    def _step(x, y):
        return T.sum(x*y, axis=1) / (T.sqrt(T.sum(x*x, axis=1) * \
                                            T.sum(y*y, axis=1))
                                     + np.float(1e-7))
    similarity, updates = theano.scan(fn = _step,
                                      sequences = [feat_seq],
                                      outputs_info = None,
                                      non_sequences = [feat],
                                      n_steps = feat_seq.shape[0])
    return similarity


def build_model(shared_params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    batch_size = options['batch_size']
    n_dim = options['n_dim']

    w_emb = shared_params['w_emb']

    dropout = theano.shared(numpy.float32(0.))
    image_feat = T.ftensor3('image_feat')
    # batch_size x T
    input_idx = T.imatrix('input_idx')
    input_mask = T.matrix('input_mask')
    # label is the TRUE label
    label = T.ivector('label')

    empty_word = theano.shared(value=np.zeros((1, options['n_emb']),
                                              dtype='float32'),
                               name='empty_word')
    w_emb_extend = T.concatenate([empty_word, shared_params['w_emb']],
                                 axis=0)
    input_emb = w_emb_extend[input_idx]

    # a trick here, set the maxpool_h/w to be large
    # maxpool_shape = (options['maxpool_h'], options['maxpool_w'])

    # turn those appending words into zeros
    # batch_size x T x n_emb
    input_emb = input_emb * input_mask[:, :, None]
    if options['sent_drop']:
        input_emb = dropout_layer(input_emb, dropout, trng, drop_ratio)

    if options['use_unigram_conv']:
        unigram_conv_feat = fflayer(shared_params, input_emb, options,
                                    prefix='conv_unigram',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        unigram_pool_feat = unigram_conv_feat.max(axis=1)
    if options['use_bigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-1],
                             T.arange(input_emb.shape[1])[1:]]).reshape((2, input_emb.shape[1] - 1)).transpose().flatten()
        bigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 1,
                                                      2 * input_emb.shape[2]))
        bigram_conv_feat = fflayer(shared_params, bigram_emb,
                                   options, prefix='conv_bigram',
                                   act_func=options.get('sent_conv_act', 'tanh'))
        bigram_pool_feat = bigram_conv_feat.max(axis=1)
    if options['use_trigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-2],
                             T.arange(input_emb.shape[1])[1:-1],
                             T.arange(input_emb.shape[1])[2:]]).reshape((3, input_emb.shape[1] - 2)).transpose().flatten()
        trigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 2,
                                                      3 * input_emb.shape[2]))
        trigram_conv_feat = fflayer(shared_params, trigram_emb,
                                    options, prefix='conv_trigram',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        trigram_pool_feat = trigram_conv_feat.max(axis=1)  #

    pool_feat = T.concatenate([unigram_pool_feat,
                               bigram_pool_feat,
                               trigram_pool_feat], axis=1)

    image_feat_down = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp',
                              act_func=options.get('image_mlp_act',
                                                   'tanh'))
    if options.get('use_before_attention_drop', False):
        image_feat_down = dropout_layer(image_feat_down, dropout, trng, drop_ratio)
        pool_feat = dropout_layer(pool_feat, dropout, trng, drop_ratio)

    # attention model begins here
    # first layer attention model
    image_feat_attention_1 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_1',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    pool_feat_attention_1 = fflayer(shared_params, pool_feat, options,
                                    prefix='sent_att_mlp_1',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    combined_feat_attention_1 = image_feat_attention_1 + \
                                pool_feat_attention_1[:, None, :]
    if options['use_attention_drop']:
        combined_feat_attention_1 = dropout_layer(combined_feat_attention_1,
                                                  dropout, trng, drop_ratio)

    combined_feat_attention_1 = fflayer(shared_params,
                                        combined_feat_attention_1, options,
                                        prefix='combined_att_mlp_1',
                                        act_func=options.get(
                                            'combined_att_mlp_act',
                                            'tanh'))
    prob_attention_1 = T.nnet.softmax(combined_feat_attention_1[:, :, 0])

    image_feat_ave_1 = (prob_attention_1[:, :, None] * image_feat_down).sum(axis=1)

    combined_hidden_1 = image_feat_ave_1 + pool_feat
    # second layer attention model

    image_feat_attention_2 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_2',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    pool_feat_attention_2 = fflayer(shared_params, combined_hidden_1, options,
                                    prefix='sent_att_mlp_2',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    combined_feat_attention_2 = image_feat_attention_2 + \
                                pool_feat_attention_2[:, None, :]
    if options['use_attention_drop']:
        combined_feat_attention_2 = dropout_layer(combined_feat_attention_2,
                                                  dropout, trng, drop_ratio)

    combined_feat_attention_2 = fflayer(shared_params,
                                        combined_feat_attention_2, options,
                                        prefix='combined_att_mlp_2',
                                        act_func=options.get(
                                            'combined_att_mlp_act', 'tanh'))
    prob_attention_2 = T.nnet.softmax(combined_feat_attention_2[:, :, 0])

    image_feat_ave_2 = (prob_attention_2[:, :, None] * image_feat_down).sum(axis=1)

    if options.get('use_final_image_feat_only', False):
        combined_hidden = image_feat_ave_2 + pool_feat
    else:
        combined_hidden = image_feat_ave_2 + combined_hidden_1


    for i in range(options['combined_num_mlp']):
        if options.get('combined_mlp_drop_%d'%(i), False):
            combined_hidden = dropout_layer(combined_hidden, dropout, trng,
                                            drop_ratio)
        if i == options['combined_num_mlp'] - 1:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='combined_mlp_%d'%(i),
                                      act_func='linear')
        else:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='combined_mlp_%d'%(i),
                                      act_func=options.get('combined_mlp_act_%d'%(i),
                                                           'tanh'))

    # drop the image output
    prob = T.nnet.softmax(combined_hidden)
    prob_y = prob[T.arange(prob.shape[0]), label]
    pred_label = T.argmax(prob, axis=1)
    # sum or mean?
    cost = -T.mean(T.log(prob_y))
    accu = T.mean(T.eq(pred_label, label))

    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu
    return image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, pred_label, \
        prob_attention_1, prob_attention_2

    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu, pred_label, \
        # image_feat_down, pool_feat

    # for debug
    # return image_feat, input_idx, input_mask, label, dropout, cost, accu, \
        # input_emb, bigram_emb, trigram_emb, trigram_pool_feat, pool_feat
