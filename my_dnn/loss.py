import numpy as np
from my_dnn.utils import one_hot_encoding

class Loss():
    def __init__(self):
        pass
    def __call__(self):
        pass
    def grad(self):
        pass

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def grad(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[1]
    
class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip values to avoid division by zero
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[1]) 

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
    
    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip values to avoid division by zero
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[1])
    
class SparseCategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        y_true = one_hot_encoding(y_true, y_pred.shape[0])
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return - np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
    
    def grad(self, y_true, y_pred):
        y_true = one_hot_encoding(y_true, y_pred.shape[0])
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[1])
    
# if __name__ == "__main__":
#     y_true = np.array([0, 1]).T
#     y_true_one_hot = np.eye(2)[y_true].T
#     print(y_true)
#     print(y_true_one_hot)
#     AL_bin = np.array([0.05, 0.8]).T
#     AL = np.array([[0.95, 0.05], [0.2, 0.8]]).T
#     BinaryCrossEntropy = BinaryCrossEntropy()
#     CategoricalCrossEntropy = CategoricalCrossEntropy()
#     SparseCategoricalCrossEntropy = SparseCategoricalCrossEntropy()
#     print(BinaryCrossEntropy(y_true, AL_bin))
#     print(CategoricalCrossEntropy(y_true_one_hot, AL))
#     print(BinaryCrossEntropy.grad(y_true, AL_bin))
#     print(CategoricalCrossEntropy.grad(y_true_one_hot, AL))
#     print(SparseCategoricalCrossEntropy(y_true, AL))
#     print(SparseCategoricalCrossEntropy.grad(y_true, AL))