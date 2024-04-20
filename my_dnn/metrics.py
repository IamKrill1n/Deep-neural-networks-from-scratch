import numpy as np
from my_dnn.utils import convert

class Metrics():
    def __init__(self):
        pass
    def __call__(self):
        pass

class MSE(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
class MAE(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

class RMSE(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

class Accuracy(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
class CategoricalAccuracy(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        y_true = convert(y_true, 'softmax')
        y_pred = convert(y_pred, 'softmax')
        return np.mean(y_true == y_pred)
    
class SparseCategoricalAccuracy(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        y_pred = convert(y_pred, 'softmax')
        return np.mean(y_true == y_pred)
    
