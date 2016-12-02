#!/usr/bin/env python

import theano
import theano.tensor as T
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
    elif options['init_type'] == 'glorot uniform':
        low = -1. * numpy.sqrt(6.0 / (n + d))
        high = 1. * numpy.sqrt(6.0 / (n + d))
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
    params['w_emb'] = ((numpy.random.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)

    params = init_fflayer(params, n_image_feat, n_dim, options,
                          prefix='image_mlp')

    # attention model based parameters
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='image_att_mlp_1')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='sent_att_mlp_1')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_1')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='image_att_mlp_2')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='sent_att_mlp_2')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_2')


    # params for sentence image mlp
    for i in range(options['combined_num_mlp']):
        if i == 0 and options['combined_num_mlp'] == 1:
            params = init_fflayer(params, n_dim, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == 0 and options['combined_num_mlp'] != 1:
            params = init_fflayer(params, n_dim, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == options['combined_num_mlp'] - 1 :
            params = init_fflayer(params, n_common_feat, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        else:
            params = init_fflayer(params, n_common_feat, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))

    # lstm layer
    params = init_lstm_layer(params, n_emb, n_dim, options, prefix='sent_lstm')

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

def init_lstm_layer(params, nin, ndim, options, prefix='lstm'):
    ''' initializt lstm layer
    '''
    params[prefix + '_w_x'] = init_weight(nin, 4 * ndim, options)
    # use svd trick to initializ
    if options['init_lstm_svd']:
        params[prefix + '_w_h'] = np.concatenate([ortho_weight(ndim),
                                                  ortho_weight(ndim),
                                                  ortho_weight(ndim),
                                                  ortho_weight(ndim)],
                                                 axis=1)
    else:
        params[prefix + '_w_h'] = init_weight(ndim, 4 * ndim, options)
    params[prefix + '_b_h'] = np.zeros(4 * ndim, dtype='float32')
    # set forget bias to be positive
    params[prefix + '_b_h'][ndim : 2*ndim] = np.float32(options.get('forget_bias', 0))
    return params

def lstm_layer(shared_params, x, mask, h_0, c_0, options, prefix='lstm'):
    ''' lstm layer:
    :param shared_params: shared parameters
    :param x: input, T x batch_size x n_emb
    :param mask: mask for x, T x batch_size
    '''
    # batch_size = optins['batch_size']
    n_dim = options['n_dim']
    # weight matrix for x, n_emb x 4*n_dim (ifoc)
    lstm_w_x = shared_params[prefix + '_w_x']
    # weight matrix for h, n_dim x 4*n_dim
    lstm_w_h = shared_params[prefix + '_w_h']
    lstm_b_h = shared_params[prefix + '_b_h']

    def recurrent(x_t, mask_t, h_tm1, c_tm1):
        ifoc = T.dot(x_t, lstm_w_x) + T.dot(h_tm1, lstm_w_h) + lstm_b_h
        # 0:3*n_dim: input forget and output gate
        i_gate = T.nnet.sigmoid(ifoc[:, 0 : n_dim])
        f_gate = T.nnet.sigmoid(ifoc[:, n_dim : 2*n_dim])
        o_gate = T.nnet.sigmoid(ifoc[:, 2*n_dim : 3*n_dim])
        # 3*n_dim : 4*n_dim c_temp
        c_temp = T.tanh(ifoc[:, 3*n_dim : 4*n_dim])
        # c_t = input_gate * c_temp + forget_gate * c_tm1
        c_t = i_gate * c_temp + f_gate * c_tm1

        if options['use_tanh']:
            h_t = o_gate * T.tanh(c_t)
        else:
            h_t = o_gate * c_t

        # if mask = 0, then keep the previous c and h
        h_t = mask_t[:, None] * h_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * h_tm1
        c_t = mask_t[:, None] * c_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * c_tm1

        return h_t, c_t

    [h, c], updates = theano.scan(fn = recurrent,
                                  sequences = [x, mask],
                                  outputs_info = [h_0[:x.shape[1]],
                                                  c_0[:x.shape[1]]],
                                  n_steps = x.shape[0])
    return h, c

def lstm_append_layer_fast(shared_params, x, mask, h_0, c_0, options,
                           prefix='lstm'):
    ''' lstm append layer fast: the h_0 and c_0 is not updated during computation
    :param shared_params: shared parameters
    :param x: input, T x batch_size x n_emb
    :param mask: mask for x, T x batch_size
    '''
    n_dim = options['n_dim']
    # weight matrix for x, n_emb x 4*n_dim (ifoc)
    lstm_w_x = shared_params[prefix + '_w_x']
    # weight matrix for h, n_dim x 4*n_dim
    lstm_w_h = shared_params[prefix + '_w_h']
    lstm_b_h = shared_params[prefix + '_b_h']
    # T x batch_size x dim
    ifoc = T.dot(x, lstm_w_x) + T.dot(h_0, lstm_w_h) + lstm_b_h
    # 0:3*n_dim: input forget and output gate
    i_gate = T.nnet.sigmoid(ifoc[:, :, 0 : n_dim])
    f_gate = T.nnet.sigmoid(ifoc[:, :, n_dim : 2*n_dim])
    o_gate = T.nnet.sigmoid(ifoc[:, :, 2*n_dim : 3*n_dim])
    # 3*n_dim : 4*n_dim c_temp
    c_temp = T.tanh(ifoc[:, :, 3*n_dim : 4*n_dim])
    # c_t = input_gate * c_temp + forget_gate * c_tm1
    c_t = i_gate * c_temp + f_gate * c_0

    if options['use_tanh']:
        h_t = o_gate * T.tanh(c_t)
    else:
        h_t = o_gate * c_t

    return h_t, c_t



def lstm_append_layer(shared_params, x, mask, h_0, c_0, options, prefix='lstm'):
    ''' lstm append layer: the h_0 and c_0 is not updated during computation
    :param shared_params: shared parameters
    :param x: input, T x batch_size x n_emb
    :param mask: mask for x, T x batch_size
    '''
    n_dim = options['n_dim']
    # weight matrix for x, n_emb x 4*n_dim (ifoc)
    lstm_w_x = shared_params[prefix + '_w_x']
    # weight matrix for h, n_dim x 4*n_dim
    lstm_w_h = shared_params[prefix + '_w_h']
    lstm_b_h = shared_params[prefix + '_b_h']

    def recurrent(x_t, mask_t, h_0, c_0):
        ifoc = T.dot(x_t, lstm_w_x) + T.dot(h_0, lstm_w_h) + lstm_b_h
        # 0:3*n_dim: input forget and output gate
        i_gate = T.nnet.sigmoid(ifoc[:, 0 : n_dim])
        f_gate = T.nnet.sigmoid(ifoc[:, n_dim : 2*n_dim])
        o_gate = T.nnet.sigmoid(ifoc[:, 2*n_dim : 3*n_dim])
        # 3*n_dim : 4*n_dim c_temp
        c_temp = T.tanh(ifoc[:, 3*n_dim : 4*n_dim])
        # c_t = input_gate * c_temp + forget_gate * c_tm1
        c_t = i_gate * c_temp + f_gate * c_0

        if options['use_tanh']:
            h_t = o_gate * T.tanh(c_t)
        else:
            h_t = o_gate * c_t

        # if mask = 0, then keep the previous c and h
        h_t = mask_t[:, None] * h_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * h_0
        c_t = mask_t[:, None] * c_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * c_0

        return h_t, c_t

    [h, c], updates = theano.scan(fn = recurrent,
                                  sequences = [x, mask],
                                  outputs_info = None,
                                  non_sequences = [h_0[:x.shape[1]], c_0[:x.shape[1]]],
                                  n_steps = x.shape[0])
    return h, c

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
    input_emb = w_emb_extend[input_idx]

    # get the transformed image feature
    h_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float32'))
    c_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float32'))

    if options['sent_drop']:
        input_emb = dropout_layer(input_emb, dropout, trng, drop_ratio)

    h_encode, c_encode = lstm_layer(shared_params, input_emb, input_mask,
                                    h_0, c_0, options, prefix='sent_lstm')
    # pick the last one as encoder
    h_encode = h_encode[-1]
    image_feat_down = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp',
                              act_func=options.get('image_mlp_act',
                                                   'tanh'))

    image_feat_attention_1 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_1',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    h_encode_attention_1 = fflayer(shared_params, h_encode, options,
                                   prefix='sent_att_mlp_1',
                                   act_func=options.get('sent_att_mlp_act',
                                                        'tanh'))  #
    combined_feat_attention_1 = image_feat_attention_1 + \
                                h_encode_attention_1[:, None, :]
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

    combined_hidden_1 = image_feat_ave_1 + h_encode

    # second layer attention model
    image_feat_attention_2 = fflayer(shared_params, image_feat_down, options,
                                     prefix='image_att_mlp_2',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    h_encode_attention_2 = fflayer(shared_params, combined_hidden_1, options,
                                   prefix='sent_att_mlp_2',
                                   act_func=options.get('sent_att_mlp_act',
                                                        'tanh'))
    combined_feat_attention_2 = image_feat_attention_2 + \
                                h_encode_attention_2[:, None, :]
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
        combined_hidden = image_feat_ave_2 + h_encode
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

    return image_feat, input_idx, input_mask, \
        label, dropout, cost, accu, pred_label
    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu, pred_label

        # h_encode, c_encode, h_decode, c_decode
