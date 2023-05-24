# https://keras-cn.readthedocs.io/en/latest/other/application/
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Flatten, Dense, Conv2D, MaxPooling2D)
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import argparse
import json


def vgg16_model(weights_path):
    # this modle totaly has 22 layers with polling 