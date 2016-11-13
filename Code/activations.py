import theano.tensor as T

def tanh(x):
    return T.tanh(x)

def relu(x):
    return T.maximum(x, np.float32(0.))

def linear(x):
    return x