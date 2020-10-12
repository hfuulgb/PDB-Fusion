# -*- coding: utf-8 -*-
import os
from keras import layers, models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, SeparableConv1D
from keras.layers import LSTM, Bidirectional
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
)
from keras.optimizers import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pdb
import utils.tools as utils
from datetime import datetime
from keras.applications import ResNet50


def get_multiscale_cnn_network():
    # 3(2)x16
    # 3,5,7(2) x16
    inputs = layers.Input(shape=(1000, 20))

    x0 = Convolution1D(
        64,
        7,
        strides=1,
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1d_0",
    )(inputs)
    x1 = layers.BatchNormalization(axis=2, name="conv1d_bn0")(x0)
    x = layers.Activation("relu")(x1)
    x = layers.MaxPool1D(3)(x)

    x11 = Convolution1D(
        64,
        7,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="x11_conv",
    )(x)
    x11 = layers.BatchNormalization()(x11)
    x11 = layers.Activation("relu")(x11)
    x11 = layers.MaxPool1D(3)(x11)
    x12 = layers.Dropout(0.1)(x11)

    x31 = Convolution1D(
        64,
        11,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="x31_conv",
    )(x)
    x31 = layers.BatchNormalization()(x31)
    x31 = layers.Activation("relu")(x31)
    x31 = layers.MaxPool1D(3)(x31)
    x32 = layers.Dropout(0.1)(x31)

    x4 = layers.concatenate([x12, x32])
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation("relu")(x4)
    x = Convolution1D(
        64,
        3,
        strides=1,
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1d_2",
    )(x4)
    x = layers.BatchNormalization(axis=2, name="conv1d_bn2")(x4)
    x = layers.Activation("relu")(x)
    # x = layers.Bidirectional(LSTM(16, return_sequences=True),
    #                          name="bidirectional_31")(x)
    x = layers.Flatten()(x4)
    x = layers.Dense(256, activation="relu")(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2, activation="sigmoid", name="dense_12")(x)

    model = models.Model(inputs, x, name="multiple_cnn")

    return model


def get_cnn_network_2layer():
    inputs = layers.Input(shape=(140, 5), name="input_6")
    x = Convolution1D(
        128,
        5,
        strides=1,
        padding="valid",
        kernel_initializer="random_uniform",
        activation="relu",
        name="conv1d_11",
    )(inputs)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_11")(x)
    x = layers.Dropout(0.1, name="dropout_21")(x)

    x = Convolution1D(
        128,
        5,
        strides=1,
        padding="valid",
        kernel_initializer="random_uniform",
        activation="relu",
        name="conv1d_12",
    )(x)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_12")(x)
    x = layers.Dropout(0.1, name="dropout_22")(x)

    # x = layers.Bidirectional(LSTM(16, return_sequences=True),
    #                          name="bidirectional_6")(x)
    x = layers.Dropout(0.2, name="dropout_23")(x)
    x = layers.Flatten(name="flatten_6")(x)
    x = layers.Dense(64, activation="relu", name="dense_11")(x)
    x = layers.Dropout(0.1, name="dropout_24")(x)
    x = layers.Dense(2, activation="sigmoid", name="dense_12")(x)

    model = models.Model(inputs, x, name="hs3d_model")
    return model
