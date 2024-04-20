import numpy as np

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    A = expZ / expZ.sum(axis=0, keepdims=True)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
     
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dA, cache):
    Z = cache
    s = np.exp(Z - np.max(Z)) / np.exp(Z - np.max(Z)).sum(axis=0, keepdims=True)
    dZ = dA * s * (1 - s)
    return dZ

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == 'linear':
        A, activation_cache = Z, Z
    cache = (linear_cache, activation_cache)
    return A, cache

def initialize_parameters_he(layer_dims, seed = 42):
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1        # number of layers in the network
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(np.divide(2, layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    elif activation == 'linear':
        dZ = dA
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def random_mini_batches(X, y, mini_batches_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    permu = np.random.permutation(m)
    X_shuffled = X[:, permu]
    y_shuffled = y[:, permu]
    mini_batches = []
    num_complete_batches = m // mini_batches_size
    for i in range(num_complete_batches):
        mini_batch_X = X_shuffled[:, i * mini_batches_size : (i + 1) * mini_batches_size]
        mini_batch_y = y_shuffled[:, i * mini_batches_size : (i + 1) * mini_batches_size]
        mini_batches.append((mini_batch_X, mini_batch_y))
    if m % mini_batches_size != 0:
        mini_batch_X = X_shuffled[:, num_complete_batches * mini_batches_size : ]
        mini_batch_y = y_shuffled[:, num_complete_batches * mini_batches_size : ]
        mini_batches.append((mini_batch_X, mini_batch_y))
    return mini_batches

def one_hot_encoding(y, num_classes):
    return np.eye(num_classes)[y].T.squeeze()

def convert(AL, activation):
    if activation == 'sigmoid':
        return AL > 0.5
    elif activation == 'softmax':
        return np.argmax(AL, axis = 0, keepdims = True)
    else:
        return AL
    
if __name__ == "__main__":
    print(one_hot_encoding(np.array([[0, 1, 2], [2, 1, 0]]), 3))
#     y_true = [0, 1, 2]
#     AL = [[0.7, 0.2, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.5]]
#     BinaryCrossEntropy = BinaryCrossEntropy()
#     CategoricalCrossEntropy = CategoricalCrossEntropy()
#     # print(BinaryCrossEntropy(y_true, AL))
#     print(CategoricalCrossEntropy(one_hot_encoding(y_true, 3), AL))
#     print(convert(AL, 'softmax'))