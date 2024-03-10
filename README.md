# Deep-neural-networks-from-scratch

Simple Dnn from scratch in Python

## Installation

To use this package, you need to have `numpy` installed. If you don't have `numpy` installed, you can install it using pip:
```
pip install numpy
```

Clone repo
```
git clone https://github.com/IamKrill1n/Deep-neural-networks-from-scratch.git
```

## Usage

Regression example
```
from my_dnn import model, optimizers, loss, metrics
# layer_dims = [number_of_feature_X, hidden_layer1, hidden_layer2, ..., hidden_layerL-1, output_layer]
my_model = model.SimpleDnn(layer_dims=[X_train.shape[1], 32, 32, 16, 1], output_activation='relu')
my_model.compile(optimizer = optimizers.RMSprop(), loss=loss.MSE(), metrics=metrics.RMSE())
# X_train must be of shape (number_of_feature, number_of_examples)
# y_train must be of shape (1, number_of_examples)
my_model.train(X_train, y_train, epochs=50, batch_size=64, verbose=0)
# X_test must be of shape (number_of_feature, number_of_examples)
y_pred= my_model.predict(X_test)
```

Classification example
```
from my_dnn import model, optimizers, loss, metrics
my_model = model.SimpleDnn(layer_dims = [X_train.shape[1], 32, 32, 16, number_of_class], output_activation = 'softmax')
my_model.compile(optimizer = optimizers.Adam(), loss = loss.CategoricalCrossEntropy(), metrics = metrics.Accuracy())
my_model.train(X_train, y_train, epochs = 50, batch_size = 64, verbose = 0)
y_pred = my_model.predict(X_test)
```

## Reference
Inspired by Coursera Deep learning Specialization https://www.coursera.org/specializations/deep-learning