
import numpy as np
import theano
from collections import OrderedDict
from theano import config

floatX = config.floatX
def ortho_weight(ndim):
	"""
	Random orthogonal weights, we take
	the right matrix in the SVD.

	Remember in SVD, u has the same # rows as W
	and v has the same # of cols as W. So we
	are ensuring that the rows are
	orthogonal.
	"""
	W = np.random.randn(ndim, ndim)
	u, _, _ = np.linalg.svd(W)
	return u.astype('float32')

def init_weight(n, d, options):
	''' initialize weight matrix
	options['init_type'] determines
	gaussian or uniform initlizaiton or glorot uniform
	'''
	if options['init_type'] == 'gaussian':
		return (np.random.randn(n, d).astype(floatX)) * options['std']
	elif options['init_type'] == 'uniform':
		# [-range, range]
		return ((np.random.rand(n, d) * 2 - 1) * \
					options['range']).astype(floatX)	
	elif options['init_type'] == 'glorot uniform':
		low = -1. * np.sqrt(6.0 / (n + d))
		high = 1. * np.sqrt(6.0 / (n + d))
		return np.random.uniform(low,high,(n,d)).astype(floatX)


def init_fflayer(params, nin, nout, options, prefix='ff'):
	''' initialize ff layer
	'''
	params[prefix + '_w'] = init_weight(nin, nout, options)
	params[prefix + '_b'] = np.zeros(nout, dtype='float32')
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
	params[prefix + '_b_h'] = np.zeros(4 * ndim, dtype='float32')
	# set forget bias to be positive
	params[prefix + '_b_h'][ndim : 2*ndim] = np.float32(options.get('forget_bias', 0))
	return params

def init_wbw_att_layer(params, nin, ndim, options, prefix='wbw_attention'):
	''' initialize Word by Word layer
	'''
	params[prefix + '_w_y'] = init_weight(ndim,ndim,options)
	params[prefix + '_w_h'] = init_weight(ndim,ndim,options)
	params[prefix + '_w_r'] = init_weight(ndim,ndim,options)
	params[prefix + '_w_alpha'] = init_weight(ndim,1,options)
	params[prefix + '_w_t'] = init_weight(ndim,ndim,options)
	return params
