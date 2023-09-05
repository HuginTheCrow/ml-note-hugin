# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"
"""
对一个学习算法来说，最重要的有四个要素：
- 数据（训练数据和测试数据）
- 模型（用于预测或分类）
- 代价函数（评价当前参数的效果，对其求导可以计算梯度）
- 优化器（优化代价函数的参数，执行梯度下降）
Beter, 20170628
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  #  用于数据缩放


housing = fetch_california_housing()
m, n = housing.data.shape  # m是样本数，n是特征的数量
print(m, n)
# Gradient Descent requires scaling the feature vectors first
# X的缩放对后面的训练过程影响非常大，经过缩放的数据经过很少的迭代次数就可以收敛，学习率可以设得很大
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.on