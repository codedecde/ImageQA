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
    return u.astype('float64')


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

    params = init_fflayer(params, n_dim, n_dim, options,
                          prefix='W_p')
    params = init_fflayer(params, n_dim, n_dim, options,
                          prefix='W_x')
    params = init_fflayer(params, n_dim, n_output, options,
                          prefix='W_h_star')

    # attention model based parameters
    # params = init_fflayer(params, n_dim, n_attention, options,
    #                       prefix='image_att_mlp_1')

    # params = init_fflayer(params, n_dim, n_attention, options,
    #                       prefix='image_att_mlp_2')
    # params = init_fflayer(params, n_dim, n_attention, options,
    #                       prefix='sent_att_mlp_2')
    # params = init_fflayer(params, n_attention, 1, options,
    #                       prefix='combined_att_mlp_2')

    # params for sentence image mlp
    # for i in range(options['combined_num_mlp']):
    #     if i == 0 and options['combined_num_mlp'] == 1:
    #         params = init_fflayer(params, n_dim, n_output,
    #                               options, prefix='combined_mlp_%d'%(i))
    #     elif i == 0 and options['combined_num_mlp'] != 1:
    #         params = init_fflayer(params, n_dim, n_common_feat,
    #                               options, prefix='combined_mlp_%d'%(i))
    #     elif i == options['combined_num_mlp'] - 1 :
    #         params = init_fflayer(params, n_common_feat, n_output,
    #                               options, prefix='combined_mlp_%d'%(i))
    #     else:
    #         params = init_fflayer(params, n_common_feat, n_common_feat,
    #                               options, prefix='combined_mlp_%d'%(i))

    # lstm layer
    params = init_lstm_layer(params, n_emb, n_dim, options, prefix='sent_lstm')

    # wbw attention layer
    params = init_wbw_att_layer(params, n_image_feat, n_dim, options)

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
    return T.maximum(x, np.float64(0.))

def linear(x):
    return x

def init_fflayer(params, nin, nout, options, prefix='ff'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options)
    params[prefix + '_b'] = np.zeros(nout, dtype='float64')
    return params

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh', nobias=False):
    ''' fflayer: multiply weight then add bias
    '''
    if nobias:
        return eval(act_func)(T.dot(x, shared_params[prefix + '_w']))
    else:
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
                       / (numpy.float64(1.0) - drop_ratio)),
                      x)
    return x_drop

def init_wbw_att_layer(params, nin, ndim, options, prefix='wbw_attention'):

    params[prefix + '_w_y'] = init_weight(nin, ndim, options)
    params[prefix + '_w_h'] = init_weight(ndim, ndim, options)
    params[prefix + '_w_r'] = init_weight(ndim, ndim, options)
    params[prefix + '_w_alpha'] = init_weight(ndim, 1, options)
    params[prefix + '_w_t'] = init_weight(ndim, ndim, options)

    return params

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
    params[prefix + '_b_h'] = np.zeros(4 * ndim, dtype='float64')
    # set forget bias to be positive
    params[prefix + '_b_h'][ndim : 2*ndim] = np.float64(options.get('forget_bias', 0))
    return params

def wbw_attention_layer(shared_params, image, question, mask, r_0, options, prefix='wbw_attention'):

    n_dim = options['n_dim']
    n_image_feat = options['n_image_feat']

    wbw_w_y = shared_params[prefix + '_w_y']
    wbw_w_h = shared_params[prefix + '_w_h']
    wbw_w_r = shared_params[prefix + '_w_r']
    wbw_w_alpha = shared_params[prefix + '_w_alpha']
    wbw_w_t = shared_params[prefix + '_w_t']

    def recurrent(h_t, mask_t, r_tm1, Y):

        tempp = T.dot(h_t, wbw_w_h) + T.dot(r_tm1, wbw_w_r)
        Mt = tanh(T.dot(Y, wbw_w_y) + tempp[:,None,:])
        WMt = T.dot(Mt, wbw_w_alpha)
        alpha_t = T.nnet.softmax(WMt[:, :, 0])
        r_t = (alpha_t[:, :, None] * Y).sum(axis=1) + tanh(T.dot(r_tm1, wbw_w_t))

        r_t = mask_t[:, None] * r_t + (numpy.float64(1.0) - mask_t[:, None]) * r_tm1

        return r_t

    [r], updates = theano.scan(fn = recurrent,
                                  sequences = [question, mask],
                                  outputs_info = [r_0[:question.shape[1]]],
                                  n_steps = question.shape[0],
                                  non_sequences = image)
    return r


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
              (numpy.float64(1.0) - mask_t[:, None]) * h_tm1
        c_t = mask_t[:, None] * c_t + \
              (numpy.float64(1.0) - mask_t[:, None]) * c_tm1

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
              (numpy.float64(1.0) - mask_t[:, None]) * h_0
        c_t = mask_t[:, None] * c_t + \
              (numpy.float64(1.0) - mask_t[:, None]) * c_0

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

    dropout = theano.shared(numpy.float64(0.))
    image_feat = T.ftensor3('image_feat')
    # T x batch_size
    input_idx = T.imatrix('input_idx')
    input_mask = T.matrix('input_mask')
    # label is the TRUE label
    label = T.ivector('label')

    empty_word = theano.shared(value=np.zeros((1, options['n_emb']),
                                              dtype='float64'),
                               name='empty_word')
    w_emb_extend = T.concatenate([empty_word, shared_params['w_emb']],
                                 axis=0)
    input_emb = w_emb_extend[input_idx]

    # get the transformed image feature
    h_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))
    c_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))

    if options['sent_drop']:
        input_emb = dropout_layer(input_emb, dropout, trng, drop_ratio)

    h_encode, c_encode = lstm_layer(shared_params, input_emb, input_mask,
                                    h_0, c_0, options, prefix='sent_lstm')
    # pick the last one as encoder

    r_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))
    r = lstm_layer(shared_params, h_encode, input_mask, h_0, r_0, options, prefix='sent_lstm')

    r = r[-1]

    Wpr = fflayer(shared_params,
                        h_encode , options,
                        prefix='W_p',
                        nobias=True,
                        act_func='linear')

    Wxh = fflayer(shared_params,
                        h_encode[-1], options,
                        prefix='W_x',
                        nobias=True,
                        act_func='linear')

    ws = T.nnet.softmax(Wpr)

    h_star = tanh(Wpr + Wxh)

    if options.get('final_dropout', False):
        h_star = dropout_layer(h_star, dropout, trng,
                                        drop_ratio)

    h_star = fflayer(shared_params, h_star, options,
                              prefix='W_h_star',
                              act_func='tanh')

    # drop the image output
    prob = T.nnet.softmax(h_star)
    prob_y = prob[T.arange(prob.shape[0]), label]
    pred_label = T.argmax(prob, axis=1)
    # sum or mean?
    cost = -T.mean(T.log(prob_y))
    accu = T.mean(T.eq(pred_label, label))

    return image_feat, input_idx, input_mask, \
        label, dropout, cost, accu
    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu, pred_label

        # h_encode, c_encode, h_decode, c_decode
