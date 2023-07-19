# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from public_function import self_print
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent

# MINIST dataset
# mnist = fetch_mldata('MNIST original')
custom_data_home = r'D:\github\datasets'
# URL, http://mldata.org/repository/data/download/matlab/mnist-original/
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
X, y = mnist['data'], mnist['target']
self_print('X shape & y shape')
print(X.shape)  # (70000, 784), 28*28 pixels
print(y.shape)
# print(mnist)

some_digit = X[36000]

# show digit's image
def show_digit_image(digit_features):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()
# show_digit_image(some_digit)
# print(y[36000])

# The MNIST dataset is acturally already split into a training set(the first 60,000 images)
# and a test set(the last 10,000 images)
train_num = 60000
X_train, X_test, y_train, y_test = X[:train_num], X[train_num:], y[:train_num], y