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
# URL, http://mlda