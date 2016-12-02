import numpy as np
import os
import sys
import logging
import theano
import pickle

np.random.seed(1337)
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical, accuracy
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers import *
from reader import *
from myutils import *
from datetime import datetime
from matplotlib import cm
from data_provision_att_vqa import *
from data_processing_vqa import *

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-xmaxlen', action="store", default=22, dest="xmaxlen", type=int)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-lr', action="store", default=0.003, dest="lr", type=float)
    parser.add_argument('-l2', action="store", default=0.0003, dest="l2", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "###################################"
    print "LSTM Output Dimension : ", opts.lstm_units
    print "Max Question Length   : ", opts.xmaxlen
    print "Dropout               : ", opts.dropout
    print "Num Epochs            : ", opts.epochs
    print "Batch Size            : ", opts.batch_size
    print "Learning Rate         : ", opts.lr
    print "Regularization Factor : ", opts.l2
    return opts

def get_H_i(i):  
    # get element i from time dimension
    def get_X_i(X):
        return X[:,i,:];
    return get_X_i

def get_H_n(X):
    # get last element from time dimension
    return X[:, -1, :]

def get_H_premise(X):
    # get elements 1 to L from time dimension
    xmaxlen = K.params['xmaxlen']
    return X[:, :xmaxlen, :] 

def get_H_hypothesis(X):
    # get elements L+1 to N from time dimension
    xmaxlen = K.params['xmaxlen']
    return X[:, xmaxlen:, :]  

def weighted_average_pooling(X):
    # Matrix A (BatchSize, Time, Feature) 
    # Matrix Alpha (BatchSize, Time, 1)
    # Matrix A Averaged along Time axis according to weights Alpha
    #    
    # Input X : (BatchSize, Time, Feature + 1) Formed by concatinating A and Alpha along Feature axis
    # Output Y : (BatchSize, Feature) Weighted average of A along time according to Alpha

    A = X[:,:,:-1]
    Alpha = X[:,:,-1]
    A = K.permute_dimensions(A, (0,2,1))  
    # Now A is (None,k,L) and Alpha is always (None,L,1)
    return K.T.batched_dot(A, Alpha)

def build_model(opts, verbose=False):

    # LSTM Output Dimension
    k = 2 * opts.lstm_units

    # Question Length
    L = opts.xmaxlen

    question_input_layer = Input(shape=(L,), dtype='int32', name="Question Input Layer")
    image_input_layer = Input(shape=(100352, ), dtype='float32', name="Image Input Layer")
    image_reshaped_layer = Reshape((196, 512), input_shape=(100352,))(image_input_layer)

    ## TODO : Get GLoVe matrix for the given vocabulary
    ##        or port existing indices to indices from Dictionary.txt
    # # Initial Embedding (Initialise using GloVe)
    # initEmbeddings = np.load(opts.embeddings_file_path)
    # emb_layer = Embedding(initEmbeddings.shape[0], 
    #                         initEmbeddings.shape[1],
    #                         input_length = L,
    #                         weights = [initEmbeddings],
    #                         name = "Embedding Layer") (question_input_layer)

    emb_layer = Embedding(13747, 
                            300,
                            input_length = L,
                            name = "Embedding Layer") (question_input_layer)
    emb_layer = Dropout(0.1, name="Dropout Embeddings")(emb_layer)

    # ## Masking Layer (May not be supported by downstream layers)
    # emb_layer = Masking(mask_value = 0., input_shape=(L, 300))(emb_layer)

    LSTMEncoding = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer")) (emb_layer)

    LSTMEncoding = Dropout(0.1, name="Dropout LSTM Layer")(LSTMEncoding)

    h_n = Lambda(get_H_n, output_shape=(k,), name = "h_n")(LSTMEncoding)

    Y = image_reshaped_layer
    Y = TimeDistributed(Dense(k, W_regularizer = l2(0.01)))(Y)

    h_hypo = LSTMEncoding
    h_hypo = TimeDistributed(Dense(k, W_regularizer = l2(0.01)))(h_hypo)

    ## Init Dense Weights
    alpha_init_weight = ((2.0/np.sqrt(k)) * np.random.rand(k,1)) - (1.0 / np.sqrt(k))
    alpha_init_bias = ((2.0) * np.random.rand(1,)) - (1.0)
    Tan_Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Tan_Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))
    Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))

    # GET R1, R2, R3, .. R_N
    for i in range(1, L+1):
        Wh_i = Lambda(get_H_i(i-1), output_shape=(k,))(h_hypo)

        if i == 1:
            M = Activation('tanh')(merge([RepeatVector(196)(Wh_i), Y], mode = 'sum'))
        else:
            M = Activation('tanh')(merge([RepeatVector(196)(Wh_i), Y, RepeatVector(196)(Wr)], mode = 'sum'))

        alpha = Reshape((196, 1), input_shape=(196,))(Activation("softmax")(Flatten()(TimeDistributed(Dense(1, weights=[alpha_init_weight, alpha_init_bias]), name='alpha'+str(i))(M))))

        r = Lambda(weighted_average_pooling, output_shape=(k,), name="r"+str(i))(merge([Y, alpha], mode = 'concat', concat_axis = 2))

        if i != 1:
            r = merge([r, Tan_Wr], mode='sum')

        if i != L:

            Tan_Wr = Dense(k, W_regularizer = l2(0.01),
                    activation = 'tanh',
                    name='Tan_Wr'+str(i), 
                    weights = [Tan_Wr_init_weight, Tan_Wr_init_bias])(r)
            Wr = Dense(k, W_regularizer = l2(0.01), 
                            name = 'Wr'+str(i), 
                            weights = [Wr_init_weight, Wr_init_bias])(r)


    r = Dense(k, W_regularizer = l2(0.01))(r) 
    h_n = Dense(k, W_regularizer = l2(0.01))(h_n)

    h_star = Activation('tanh')(merge([r, h_n]))

    output_layer = Dense(1000, activation='softmax', name="Output Layer")(h_star)

    model = Model(input = [image_input_layer, question_input_layer], output = output_layer)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(options.lr), metrics=['accuracy'])
    print "Model Compiled"

    return model

## Data Generator
def DataIterator(data, split, options):
    while (True):
        for image, question, answer in  data.iterate_batch(split, options.batch_size):
            question = pad_sequences(question, maxlen = options.xmaxlen, value = 0, padding = 'post')
            answer = to_categorical(answer, nb_classes = 1000)
            yield ([image, question], answer)

def compute_acc(X, Y, vocab, model, opts, filename=None):
    scores = model.predict(X, batch_size = options.batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        prediction[i][np.argmax(scores[i])] = 1.0

    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(Y, axis=1)
    acc = accuracy(tlabels, plabels)

    if filename != None:
        with open(filename, 'w') as f:
            for i in range(len(X)):
                f.write(map_to_txt(X[i],vocab)+ " : "+ str(plabels[i])+ "\n")

    return acc

def getConfig(opts):
    conf=[opts.xmaxlen,
          opts.ymaxlen,
          opts.batch_size,
          opts.lr,
          opts.lstm_units,
          opts.epochs]

    return "_".join(map(lambda x: str(x), conf))

def save_model(model,wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        yaml_string = model.to_yaml()
        open(archpath, 'w').write(yaml_string)
    else:
        with open(archpath, 'w') as f:
            f.write(model.to_json())
    model.save_weights(wtpath)


def load_model(wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        model = model_from_yaml(open(archpath).read())
    else:
        with open(archpath) as f:
            model = model_from_json(f.read())
    model.load_weights(wtpath)
    return model

class WeightSharing(Callback):
    def __init__(self, shared_group_name, num_members):
        self.shared_group_name = shared_group_name
        self.num_members = num_members

    def find_layer_by_name(self, name):
        for l in self.model.layers:
            if l.name == name:
                return l

    def on_batch_end(self, batch, logs={}):
        group = [self.shared_group_name + str(i) for i in xrange(1, self.num_members+1)]
        weights = np.mean([self.find_layer_by_name(n).get_weights()[0] for n in group], axis=0)
        biases = np.mean([self.find_layer_by_name(n).get_weights()[1] for n in group], axis=0)
        for n in group:
            self.find_layer_by_name(n).set_weights([weights, biases])

class WeightSave(Callback):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_epoch_end(self, epochs, logs={}):
        self.model.save_weights(file_path + str(epochs) +  ".weights")  

def RTE(premise, hypothesis,vocab,model):
    labels = {0:'neutral',1:'entailment',2:'contradiction'}
    p = map_to_idx(tokenize(premise),vocab)
    h = map_to_idx(tokenize(hypothesis),vocab)
    p = pad_sequences([p], maxlen=options.xmaxlen,value=vocab["pad_tok"],padding='pre')
    h = pad_sequences([h], maxlen=options.ymaxlen,value=vocab["pad_tok"],padding='post')
    sentence = concat_in_out(p,h,vocab)
    scores = model.predict(sentence,batch_size=1)
    return labels[np.argmax(scores)]

if __name__ == "__main__":

    ## Fetch Params
    options = get_params()
    setattr(K, 'params', {'xmaxlen': options.xmaxlen})
    if options.local:
        options.embeddings_file_path = 'VocabMat.npy'
        # with open('Dictionary.txt','r') as inf:
        #     vocab = eval(inf.read())
        data_path = '../HPCData/'
    else:
        options.embeddings_file_path = '/home/cse/btech/cs1130773/Code/VocabMat.npy'
        # with open('/home/cse/btech/cs1130773/Code/Dictionary.txt') as inf:
        #     vocab = eval(inf.read())
        data_path = '../../../scratch/DLProjData/'            

    ## Load Data
    with open(data_path + 'question_dict.pkl','r') as inf:
        vocab = pickle.load(inf)            
        print "Vocab Size            : ", len(vocab)

    data = DataProvisionAttVqa(data_path, 'trainval_feat.h5')

    # ## Path to stored Model
    # config_str = getConfig(options)
    # MODEL_ARCH = "/home/ee/btech/ee1130798/Code/Models/ATRarch_att" + config_str + ".yaml"
    # MODEL_WGHT = "/home/ee/btech/ee1130798/Code/Models/ATRweights_att" + config_str + ".weights"
    # MODEL_ARCH = "/Users/Shantanu/Documents/College/SemVI/COL772/Project/Code/Models/GloveEmbd/arch_att" + config_str + ".yaml"
    # MODEL_WGHT = "/Users/Shantanu/Documents/College/SemVI/COL772/Project/Code/WeightsMultiAttention/weight_on_epoch_6.weights"

    ## Build Model   
    print "###################################"
    print 'Building model'
    model = build_model(options)

    ## Train Model
    print "###################################"
    print 'Training New Model'

    # ## TO save weights at end of every epoch
    # save_weights = WeightSave("/<absolute Path to file>/bestWeights")

    history = model.fit_generator( DataIterator(data, "train", options),
                    samples_per_epoch = data.get_size("train"),
                    nb_epoch = options.epochs,
                    validation_data = DataIterator(data, "val1", options),
                    nb_val_samples = data.get_size("val1"),
                    callbacks = [WeightSharing('Tan_Wr', options.xmaxlen-1), 
                                WeightSharing('Wr', options.xmaxlen-1), 
                                WeightSharing('alpha', options.xmaxlen)])

    # train_acc = compute_acc(net_train, Z_train, vocab, model, options)
    # dev_acc   = compute_acc(net_dev, Z_dev, vocab, model, options)
    # test_acc  = compute_acc(net_test, Z_test, vocab, model, options)
    # print "Training Accuracy: ", train_acc
    # print "Dev Accuracy: ", dev_acc
    # print "Testing Accuracy: ", test_acc

