import numpy
from collections import OrderedDict
def get_configurations():
	##################
	# initialization #
	##################
	options = OrderedDict()
	# data related
	options['data_path'] = '/home/ee/btech/ee1130798/scratch/DLProjData/'
	options['feature_file'] = 'trainval_feat.h5'
	options['expt_folder'] = '/home/ee/btech/ee1130798/DL/Proj/Models'
	options['model_name'] = 'imageqa'
	options['train_split'] = 'trainval1'
	options['val_split'] = 'val2'
	options['shuffle'] = True
	options['reverse'] = True
	options['sample_answer'] = True
	
	options['num_region'] = 196
	options['region_dim'] = 512

	options['n_words'] = 13746
	options['n_output'] = 1000

	# structure options
	options['combined_num_mlp'] = 1
	options['combined_mlp_drop_0'] = True
	options['combined_mlp_act_0'] = 'linear'
	options['sent_drop'] = False
	options['use_tanh'] = False

	options['use_attention_drop'] = False

	# dimensions
	options['n_emb'] = 300 # Default 500
	options['n_dim'] = 1024
	options['n_image_feat'] = options['region_dim']
	options['n_common_feat'] = 500
	options['n_attention'] = 512
	options['embedding_file'] = '/home/ee/btech/ee1130798/DL/Proj/embedding_matrix.pkl' 
	# initialization
	options['init_type'] = 'glorot uniform'
	options['range'] = 0.01
	options['std'] = 0.01
	options['init_lstm_svd'] = False

	options['forget_bias'] = numpy.float32(1.0)

	# learning parameters
	options['optimization'] = 'adam' # choices
	options['batch_size'] = 100
	options['lr'] = numpy.float32(0.05)
	options['w_emb_lr'] = numpy.float32(80)
	options['momentum'] = numpy.float32(0.9)
	options['gamma'] = 1
	options['step'] = 10
	options['step_start'] = 100
	options['max_epochs'] = 50
	options['weight_decay'] = 0.0005
	options['decay_rate'] = numpy.float32(0.999)
	options['drop_ratio'] = numpy.float32(0.5)
	options['smooth'] = numpy.float32(1e-8)
	options['grad_clip'] = numpy.float32(0.1)

	# log params
	options['disp_interval'] = 10
	options['eval_interval'] = 1000
	options['save_interval'] = 500

	return options
