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
    """Function to define a standard CNN model
    :param n_outputs: the number of units in the last layer
    :param n_filters: base number of convolutional filters
    """
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])
    return model


@tf.function
def standard_train_step(x, y):
    with tf.GradientTape() as tape:
        # feed the images into the model
        logits = standard_classifier(x)
        # Compute the loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    # Backpropagation
    grads = tape.gradient(loss, standard_classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
    return loss


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    """ Function to calculate VAE loss given:
          an input x,
          reconstructed output x_recon,
          encoded means mu,
          encoded log of standard deviation logsigma,
          weight parameter for the latent loss kl_weight
    """
    # Define the latent loss. Note this is given in the equation for L_{KL} in the text block, and measures how closely
    # the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=