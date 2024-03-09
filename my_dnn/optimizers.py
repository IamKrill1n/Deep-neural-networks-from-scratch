import numpy as np

class Optimizer():
    def __init__(self,):
        pass

    def update_parameters(self, parameters, grads):
        pass

class GD():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.name = 'GD'

    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * grads["db" + str(l)]
        return parameters
    
class SGD():
    def __init__(self, learning_rate, beta = 0.9):
        self.learning_rate = learning_rate
        self.beta  = beta
        self.name = 'SGD'

    def initialize_velocity(self, parameters):
        L = len(parameters) // 2
        self.v = {}
        for l in range(1, L + 1):
            self.v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            self.v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            self.v["dW" + str(l)] = self.v["dW" + str(l)] * self.beta + grads["dW" + str(l)] * (1 - self.beta)
            self.v["db" + str(l)] = self.v["db" + str(l)] * self.beta + grads["db" + str(l)] * (1 - self.beta)
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * self.v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * self.v["db" + str(l)]
        return parameters
    
class RMSprop():
    def __init__(self, learning_rate, beta, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.name = 'RMSprop'

    def initialize_rmsprop(self, parameters):
        L = len(parameters) // 2
        self.s = {}
        for l in range(1, L + 1):
            self.s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            self.s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            self.s["dW" + str(l)] = self.beta * self.s["dW" + str(l)] + (1 - self.beta) * np.square(grads["dW" + str(l)])
            self.s["db" + str(l)] = self.beta * self.s["db" + str(l)] + (1 - self.beta) * np.square(grads["db" + str(l)])
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * (grads["dW" + str(l)] / np.sqrt(self.s["dW" + str(l)] + self.epsilon))
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * (grads["db" + str(l)] / np.sqrt(self.s["db" + str(l)] + self.epsilon))
        return parameters
    
class Adam():
    def __init__(self, learning_rate, beta1 = 0.9, beta2 = 0.999, t = 0, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
        self.epsilon = epsilon
        self.name = 'Adam'

    def initialize_adam(self, parameters):
        L = len(parameters) // 2
        self.v = {}
        self.s = {}
        for l in range(1, L + 1):
            self.v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
            self.v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
            self.s['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
            self.s['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)        
        
    def update_parameters_with_adam(self, parameters, grads):
        L = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L + 1):
            # update v
            self.v['dW' + str(l)] = self.beta1 * self.v['dW' + str(l)] + (1 - self.beta1) * grads['dW' + str(l)]
            self.v['db' + str(l)] = self.beta1 * self.v['db' + str(l)] + (1 - self.beta1) * grads['db' + str(l)]

            # update v corrected
            v_corrected['dW' + str(l)] = self.v['dW' + str(l)] / (1 - self.beta1 ** self.t)
            v_corrected['db' + str(l)] = self.v['db' + str(l)] / (1 - self.beta1 ** self.t)

            # update s
            self.s['dW' + str(l)] = self.beta2 * self.s['dW' + str(l)] + (1 - self.beta2) * grads['dW' + str(l)] ** 2
            self.s['db' + str(l)] = self.beta2 * self.s['db' + str(l)] + (1 - self.beta2) * grads['db' + str(l)] ** 2

            # update s corrected
            s_corrected['dW' + str(l)] = self.s['dW' + str(l)] / (1 - self. beta2 ** self.t)
            s_corrected['db' + str(l)] = self.s['db' + str(l)] / (1 - self.beta2 ** self.t)

            # update parameters
            parameters['W' + str(l)] = parameters['W' + str(l)] - self.learning_rate * v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + self.epsilon)
            parameters['b' + str(l)] = parameters['b' + str(l)] - self.learning_rate * v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + self.epsilon)
        