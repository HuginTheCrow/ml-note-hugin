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
from sklearn.model_selection import cross_val_score  # 用于训练集内部交叉验证
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler  # 用于输入数据的预处理
from sklearn.metrics import confusion_matrix  # 用于评价模型


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

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

# The MNIST dataset is acturally already split into a training set(the first 60,000 images)
# and a test set(the last 10,000 images)
train_num = 60000
X_train, X_test, y_train, y_test = X[:train_num], X[train_num:], y[:train_num], y[train_num:]

# shuffle the training set, 打乱训练样本的次序，这一步在有些时候挺有用的
# 比如样本是排过序的，shuffle后可以保证每种数字的分布是均匀的
shuffle_index = np.random.permutatio