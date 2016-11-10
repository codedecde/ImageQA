#!/usr/bin/env python

import numpy
import numpy as np
import theano
import theano.tensor as T

def sgd(shared_params, grads, options):
    '''
    grads is already the shared variable containing the gradients, we only
    need to do a accumulation and then do an updated
    '''
    momentum = options['momentum']
    # the cache here can not be reseach by outside function
    lr = T.scalar('lr')
    grad_cache = [theano.shared(p.get_value() * numpy.float32(0.),
                                name='%s_grad_cache' % k )
                  for k, p in shared_params.iteritems()]
    # update the caches
    grad_cache_update = [(g_c, g_c * momentum + g)
                         for g_c, g in zip (grad_cache, grads)]
    param_update = [(p, p - lr * options.get('%s_lr'%(k), 1.0) * g_c )
                    for k, p, g_c in zip(shared_params.keys(),
                                         shared_params.values(),
                                         grad_cache)]

    # two functions: do the grad cache updates and param_update
    f_grad_cache_update = theano.function([], [],
                                          updates = grad_cache_update,
                                          name = 'f_grad_cache_update')
    f_param_update = theano.function([lr], [],
                                     updates = param_update,
                                     name = 'f_param_update')

    return f_grad_cache_update, f_param_update


def rmsprop(shared_params, grads, options):
    decay_rate = options['decay_rate']
    smooth = options['smooth']

    lr = T.scalar('lr')
    # the cache here can not be reseach by outside function
    grad_cache = [theano.shared(p.get_value() * numpy.float32(0.),
                                name='%s_grad_cache' % k)
                  for k, p in shared_params.iteritems()]
    # update the caches
    grad_cache_update = [(g_c, g_c * decay_rate +
                          (numpy.float32(1.) - decay_rate) * g**2)
                         for g_c, g in zip (grad_cache, grads)]
    param_update = [(p, p - lr * options.get('%s_lr'%(k), 1.0) * g / T.sqrt(g_c + smooth))
                    for p, g_c, g in zip(shared_params.values(), grad_cache, grads)]

    # two functions: do the grad cache updates and param_update
    f_grad_cache_update = theano.function([], [],
                                          updates = grad_cache_update,
                                          name = 'f_grad_cache_update')
    f_param_update = theano.function([lr], [],
                                     updates = param_update,
                                     name = 'f_param_update')

    return f_grad_cache_update, f_param_update

def rmsprop_nestrov(shared_params, grads, options):
    # the parameters are fixed !!
    lr = T.scalar('lr')
    decay_rate = np.float32(0.95)
    running_grad = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name = '%s_running_grad' %k)
                    for k, p in shared_params.iteritems()]
    running_grad_sqr = [theano.shared(p.get_value() * numpy.float32(0.),
                                      name = '%s_running_grad_sqr' %k)
                        for k, p in shared_params.iteritems()]
    # delta_weight is the actual update
    delta_weight = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name = '%s_weight_update' %k)
                     for k, p in shared_params.iteritems()]

    # below is the grad cache update
    running_grad_update = [(r_g, r_g * decay_rate + (np.float32(1.0) - decay_rate) * g)
                           for r_g, g in zip(running_grad, grads)]
    running_grad_sqr_update = [(r_g_sqr, r_g_sqr * decay_rate + (np.float32(1.0) - decay_rate) * g**2)
                               for r_g_sqr, g in zip(running_grad_sqr, grads)]
    # acutual amount deducted from param
    delta_weight_update = [(d_w, np.float32(0.9) * d_w + np.float32(1e-4) * g /
                            T.sqrt(r_g_sqr - r_g**2 + np.float32(1e-4)))
                           for d_w, g, r_g, r_g_sqr in zip(delta_weight,
                                                           grads,
                                                           running_grad,
                                                           running_grad_sqr)]
    # TODO: resolve the dependency here
    # actual param update
    param_update = [(p, p - d_w_u[1])
                    for p, d_w_u in zip(shared_params.values(), delta_weight_update)]

    f_grad_cache_update = theano.function([], [],
                                          updates = running_grad_update + \
                                          running_grad_sqr_update,
                                          name = 'f_grad_cache_update')

    # lr is actually not used here
    f_param_update = theano.function([lr], [],
                                     updates = param_update +\
                                     delta_weight_update,
                                     name = 'f_param_update',
                                     on_unused_input='ignore')
    return f_grad_cache_update, f_param_update

def adagrad(shared_params, grads, options):
    lr = T.scalar('lr')
    running_grad_sqr = [theano.shared(p.get_value() * np.float32(0.),
                                      name = '%s_running_grad_sqr' %k)
                        for k, p in shared_params.iteritems()]
    running_grad_sqr_update = [(r_g_sqr, r_g_sqr + g**2)
                               for r_g_sqr, g in zip(running_grad_sqr, grads)]

    param_update = [(p, p - lr * g / T.sqrt(r_g_sqr))
                    for p, g, r_g_sqr in zip(shared_params.values(),
                                             grads, running_grad_sqr)]

    # grad cache update function
    f_grad_cache_update = theano.function([], [],
                                          updates = running_grad_sqr_update,
                                          name = 'f_grad_cache_update')
    f_param_update = theano.function([lr], [],
                                     updates = param_update,
                                     name = 'f_param_update')
    return f_grad_cache_update, f_param_update


def adam(shared_params, grads, options):
    lr = T.scalar('lr')
    running_grad = [theano.shared(p.get_value() * np.float32(0.),
                                  name = '%s_running_grad' %k)
                    for k, p in shared_params.iteritems()]
    running_grad_sqr = [theano.shared(p.get_value() * np.float32(0.),
                                      name = '%s_running_grad_sqr' %k)
                        for k, p in shared_params.iteritems()]
    t = theano.shared(numpy.float32(0.))

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    running_grad_update = [(r_g, b1 * g + (1- b1) * r_g)
                            for r_g, g in zip(running_grad, grads)]
    running_grad_sqr_update = [(r_g_sqr, b2 * g**2 + (1- b2) * r_g_sqr)
                               for r_g_sqr, g in zip(running_grad_sqr, grads)]
    time_update = [(t, t + 1)]
    f_grad_cache_update = theano.function([], [],
                                          updates = running_grad_update +
                                          running_grad_sqr_update +
                                          time_update,
                                          name = 'f_grad_cache_update')

    fix1 = 1. - b1**(t)
    fix2 = 1. - b2**(t)
    lr_t = lr0 * (T.sqrt(fix2) / fix1)
    param_update = [(p, p - lr_t * r_g / (T.sqrt(r_g_sqr) + e))
                    for p, r_g, r_g_sqr in zip(shared_params.values(),
                                               running_grad,
                                               running_grad_sqr)]

    f_update = theano.function([lr], [], updates = param_update,
                               on_unused_input='ignore')

    return f_grad_cache_update, f_update


def adadelta(shared_params, grads, options):
    lr = T.scalar('lr')
    running_up_sqr = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_running_up_sqr'%k)
                      for k, p in shared_params.iteritems()]
    running_grad_sqr = [theano.shared(p.get_value() * numpy.float32(0.),
                                      name='%s_running_grad_sqr'%k)
                        for k, p in shared_params.iteritems()]

    running_grad_sqr_update = [(r_g_sqr, 0.95 * r_g_sqr + 0.05 * (g ** 2))
                               for r_g_sqr, g in zip(running_grad_sqr, grads)]

    f_grad_cache_update = theano.function([], [],
                                          updates=running_grad_sqr_update,
                                          profile=False)

    updir = [T.sqrt(r_u_sqr + 1e-6) / T.sqrt(r_g_sqr + 1e-6) * g for g,
             r_u_sqr, r_g_sqr in zip(grads, running_up_sqr, running_grad_sqr)]
    running_up_sqr_update = [(r_u_sqr, 0.95 * r_u_sqr + 0.05 * (ud ** 2))
                             for r_u_sqr, ud in zip(running_up_sqr, updir)]
    param_upate = [(p, p - ud) for p, ud in zip(shared_params.values(), updir)]

    f_update = theano.function([lr], [],
                               updates=param_upate +
                               running_up_sqr_update,
                               on_unused_input='ignore',
                               profile=False)

    return f_grad_cache_update, f_update
