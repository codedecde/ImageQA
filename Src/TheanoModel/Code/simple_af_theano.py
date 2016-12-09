#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy
import numpy as np
import pickle as pkl
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


def init_weight(n, d, options, activation='tanh'):
    ''' initialize weight matrix
    options['init_type'] determines
    gaussian or uniform initlizaiton
    '''
    if options['init_type'] == 'gaussian' or activation == 'relu':
        return (numpy.random.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        return ((numpy.random.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)
    
    elif options['init_type'] == 'glorot uniform':
        low = -1.0 * np.sqrt(6.0/(n + d))
        high = 1.0 * np.sqrt(6.0/(n + d))
        if activation == 'sigmoid':
            low = low * 4.0
            high = high * 4.0
        return numpy.random.uniform(low,high,(n,d)).astype(floatX)

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
    n_image_feat = options['n_image_feat']
    n_common_feat = options['n_common_feat']
    n_output = options['n_output']
    n_attention = options['n_attention']

    # embedding weights
    # params['w_emb'] = init_weight(n_words, n_emb, options)
    ## use the same initialization as BOW
    # params['w_emb'] = ((numpy.random.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)
    embedding_matrix = pkl.load(open(options['embedding_file'], 'r'))[1:].astype(floatX)
    params['w_emb'] = embedding_matrix

    params = init_fflayer(params, n_image_feat, n_dim, options,
                          prefix='image_mlp')

    params = init_fflayer(params, n_dim, n_dim, options,
                          prefix='F_func', activation='relu')

    params = init_fflayer(params, 2*n_dim, 2*n_dim, options,
                          prefix='G_func', activation='relu')

    params = init_fflayer(params, 4*n_dim, n_output, options,
                          prefix='scale_to_softmax')

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

def init_fflayer(params, nin, nout, options, prefix='ff', activation='tanh'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options,activation)
    params[prefix + '_b'] = np.zeros(nout, dtype='float32')
    if activation == 'relu':
        params[prefix + '_b'] = np.ones(nout, dtype='float32')
    return params

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh'):
    ''' fflayer: multiply weight then add bias
    '''
    return eval(act_func)(T.dot(x, shared_params[prefix + '_w']) +
                          shared_params[prefix + '_b'])

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

def batchedSoftmax(x, axis=1):
    if axis == 1:
        x = x.dimshuffle((0,2,1))
    init_shape = x.shape
    x = x.reshape((init_shape[0]*init_shape[1], init_shape[2]))
    x = T.nnet.softmax(x)
    x = x.reshape(init_shape)
    if axis == 1:
        x = x.dimshuffle((0,2,1))
    return x   

def build_model(shared_params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    batch_size = options['batch_size']
    n_dim = options['n_dim']

    w_emb = shared_params['w_emb']

    dropout = theano.shared(numpy.float32(0.))
    image_feat = T.ftensor3('image_feat')
    # T x batch_size
    input_idx = T.imatrix('input_idx')
    input_mask = T.matrix('input_mask')
    # label is the TRUE label
    label = T.ivector('label')

    empty_word = theano.shared(value=np.zeros((1, options['n_emb']),
                                              dtype='float32'),
                               name='empty_word')
    w_emb_extend = T.concatenate([empty_word, shared_params['w_emb']],
                                 axis=0)


    a = w_emb_extend[input_idx] # T x bt_sz x n_dim
    b = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp',
                              act_func=options.get('image_mlp_act',
                                                   'tanh')) # bt_sz x num_regions x n_dim
    Fb = fflayer(shared_params, b, options, 
                              prefix='F_func', 
                              act_func='relu') # bt x num_region x n_dim
    Fa = fflayer(shared_params, a, options,
                                prefix='F_func',
                                act_func='relu') # T x bt_sz x n_dim
    e = T.batched_dot(Fa.dimshuffle((1,0,2)) , Fb.dimshuffle((0,2,1) ) ) # bt x T x num_regions

    alpha = batchedSoftmax(e,1) # bt x T x num_regions 
    beta = batchedSoftmax(e,2) # bt x T x num_regions


    alpha = T.batched_dot(alpha.dimshuffle((0,2,1)), a.dimshuffle((1,0,2))) # bt x num_regions x ndim
    beta = T.batched_dot(beta, b) # bt x T x ndim

    a_beta = T.concatenate([a.dimshuffle((1,0,2)), beta], axis=2) # bt x T x 2*ndim
    b_alpha = T.concatenate([b, alpha], axis=2) # bt x num_regions x 2*ndim


    G_a_beta = fflayer(shared_params, a_beta, options,
                                prefix='G_func',
                                act_func='tanh') # bt x T x 2*ndim
    G_b_alpha = fflayer(shared_params, b_alpha, options, 
                              prefix='G_func', 
                              act_func='tanh') # bt x num_regions x 2*ndim

    V1 = T.sum(G_a_beta, axis=1) # bt x 2*ndim
    V2 = T.sum(G_b_alpha, axis=1) # bt x 2*ndim

    h_star = T.concatenate([V1, V2], axis=1) # bt x 4*dim

    ## Final Dense
    combined_hidden = fflayer(shared_params, h_star, options,
                                prefix='scale_to_softmax',
                                act_func='tanh')

    # drop the image output
    prob = T.nnet.softmax(combined_hidden)
    prob_y = prob[T.arange(prob.shape[0]), label]
    pred_label = T.argmax(prob, axis=1)
    confidence = T.max(prob, axis=1)
    # sum or mean?
    cost = -T.mean(T.log(prob_y))
    accu = T.mean(T.eq(pred_label, label))

    return image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, alpha, pred_label, confidence
    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu, pred_label

        # h_encode, c_encode, h_decode, c_decode
