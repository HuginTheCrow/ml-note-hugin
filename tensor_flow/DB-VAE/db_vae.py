"""
MIT 6.S191, lab2
https://github.com/aamini/introtodeeplearning/blob/master/lab2/Part2_Debiasing.ipynb
http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf
"""

import IPython
import tensorflow as tf
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mitdeeplearning as mdl


def make_standard_classifier(n_outputs=1, n_filters=12):
    """Function to define a standard CNN m