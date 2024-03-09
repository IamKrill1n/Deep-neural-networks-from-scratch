import numpy as np

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

class Accuracy(Metrics):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
