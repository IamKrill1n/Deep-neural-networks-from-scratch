import numpy as np
from my_dnn.utils import *

class SimpleMlp:
    def __init__(self, layer_dims, output_activation="sigmoid"):
        self.layer_dims = layer_dims
        self.output_activation = output_activation
        self.parameters = initialize_parameters_he(layer_dims)

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        if self.optimizer.name == 'SGD':
            self.optimizer.initialize_velocity(self.parameters)
        elif self.optimizer.name == 'RMSprop':
            self.optimizer.initialize_rmsprop(self.parameters)
        elif self.optimizer.name == 'Adam':
            self.optimizer.initialize_adam(self.parameters)
        self.loss = loss
        self.metrics = metrics
    
    def save_weights(self, filename):
        np.savez(filename, **self.parameters)

    def load_weights(self, filename):
        self.parameters = dict(np.load(filename))

    def forward_propagation(self, X):
        caches = []
        A = X
        L = len(self.layer_dims) - 1
        for l in range(1, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], "relu")
            caches.append(cache)
        AL, cache = linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], self.output_activation)
        caches.append(cache)
        return AL, caches
    
    def backward_propagation(self, Y, AL, caches):
        dAL = self.loss.grad(Y, AL)
        L = len(caches) 
        grads = {}
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, caches[L - 1], self.output_activation)
        grads['dW' + str(L)] = dW_temp
        grads['db' + str(L)] = db_temp
        for l in range(L - 2, -1, -1):
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, caches[l], 'relu')
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp
            
        return grads
    
    def fit(self, X, y, validation_data, epochs = 100, batch_size = 32, val_batch_size = 32, verbose = 0):
        X = X.T
        y = y.reshape(1, -1)
        m = X.shape[1]
        self.history = {
            'loss': [],
            'eval': [],
            'val_loss': [],
            'val_eval': []
        }
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.T
            y_val = y_val.reshape(1, -1)

        for i in range(epochs):
            minibatches = random_mini_batches(X, y, batch_size, i)
            cost_total = 0
            eval_total = 0
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # Forward propagation
                AL, caches = self.forward_propagation(minibatch_X)
                # Compute cost
                cost = self.loss(minibatch_Y, AL)
                cost_total += cost * minibatch_X.shape[1]
                # Compute evaluation metric
                eval = self.metrics(minibatch_Y, AL)
                eval_total += eval * minibatch_X.shape[1]
                # Backward propagation
                grads = self.backward_propagation(minibatch_Y, AL, caches)
                # Update parameters
                self.parameters = self.optimizer.update_parameters(self.parameters, grads)
                
            cost_avg = cost_total / m
            eval_avg = eval_total / m
            self.history['loss'].append(cost_avg)
            self.history['eval'].append(eval_avg)

            if validation_data is not None:
                minibatches_val = random_mini_batches(X_val, y_val, val_batch_size, i)
                val_cost_total = 0
                val_eval_total = 0
                for minibatch in minibatches_val:
                    (minibatch_X, minibatch_Y) = minibatch
                    # Forward propagation
                    AL_val, _ = self.forward_propagation(minibatch_X)
                    # Compute cost
                    val_cost = self.loss(minibatch_Y, AL_val)
                    val_cost_total += val_cost * minibatch_X.shape[1]
                    # Compute evaluation metric
                    val_eval = self.metrics(minibatch_Y, AL_val)
                    val_eval_total += val_eval * minibatch_X.shape[1]

                val_cost = val_cost_total / X_val.shape[1]
                val_eval = val_eval_total / X_val.shape[1]
                self.history['val_loss'].append(val_cost)
                self.history['val_eval'].append(val_eval)
                    
            if verbose:
                print("Epoch " + str(i) + " Cost: " + str(cost_avg) + " Eval: " + str(eval_avg) + " Val Cost: " + str(val_cost) + " Val Eval: " + str(val_eval))

    def get_history(self):
        return self.history
    
    def predict(self, X):
        AL, _ = self.forward_propagation(X.T)
        return convert(AL, self.output_activation).T