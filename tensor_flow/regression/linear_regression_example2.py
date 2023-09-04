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
from sk