import numpy as np

class Loss():
    def __init__(self):
        pass
    def __call(self):
        pass
    def grad(self):
        pass

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def grad(self, y_true, y_pred):
        return y_pred - y_true
    
class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return - np.mean(np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T))
    
    def grad(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return - np.mean(np.dot(y_true, np.log(y_pred).T))
    
    def grad(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))