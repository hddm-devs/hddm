from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import argparse
#import sys
#import tempfile
#import os, glob, pickle
import tensorflow as tf
from .config import *

# from .tf_data_handler import inputs
import numpy as np
#import tqdm, time
#import matplotlib.pyplot as plt
#import math
#from textwrap import wrap


class CNNModelStruct:
    def __init__(self, trainable=True):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self, item)

    def __contains__(self, item):
        return hasattr(self, item)

    def get_size(self, input_data):
        return np.prod([int(x) for x in input_data.get_shape()[1:]])

    def build(
        self, input_data, input_shape, output_shape, train_mode=None, verbose=True
    ):
        if verbose:
            print("Building the network...")
        network_input = tf.identity(input_data, name="input")
        with tf.name_scope("reshape"):
            x_data = tf.reshape(
                network_input, [-1, input_shape[0], input_shape[1], input_shape[2]]
            )
        self.upsample1 = self.fc_layer(x_data, self.get_size(x_data), 16, "upsample1")
        if verbose:
            print(self.upsample1.get_shape())
        self.upsample2 = self.fc_layer(
            self.upsample1, self.get_size(self.upsample1), 64, "upsample2"
        )
        if verbose:
            print(self.upsample2.get_shape())
        self.upsample3 = self.fc_layer(
            self.upsample2, self.get_size(self.upsample2), 256, "upsample3"
        )
        if verbose:
            print(self.upsample3.get_shape())
        self.upsample4 = self.fc_layer(
            self.upsample3, self.get_size(self.upsample3), 1024, "upsample4"
        )
        if verbose:
            print(self.upsample4.get_shape())

        self.upsample4 = tf.expand_dims(tf.expand_dims(self.upsample4, 1), -1)

        # conv layer 1
        with tf.variable_scope("conv1"):
            self.W_conv1 = self.weight_variable([1, 5, 1, 8], var_name="wconv1")
            self.b_conv1 = self.bias_variable([8], var_name="bconv1")
            self.norm1 = tf.layers.batch_normalization(
                self.conv2d(self.upsample4, self.W_conv1, stride=[1, 2, 2, 1])
                + self.b_conv1,
                scale=True,
                center=True,
                training=train_mode,
            )
            self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
        if verbose:
            print(self.h_conv1.get_shape())

        # conv layer 2
        with tf.variable_scope("conv2"):
            self.W_conv2 = self.weight_variable([1, 5, 8, 4], var_name="wconv2")
            self.b_conv2 = self.bias_variable([4], var_name="bconv2")
            self.norm2 = tf.layers.batch_normalization(
                self.conv2d(self.h_conv1, self.W_conv2, stride=[1, 1, 1, 1])
                + self.b_conv2,
                scale=True,
                center=True,
                training=train_mode,
            )
            self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
        if verbose:
            print(self.h_conv2.get_shape())

        # conv layer 3
        with tf.variable_scope("conv3"):
            self.W_conv3 = self.weight_variable([1, 5, 4, 2], var_name="wconv3")
            self.b_conv3 = self.bias_variable([2], var_name="bconv3")
            self.norm3 = tf.layers.batch_normalization(
                self.conv2d(self.h_conv2, self.W_conv3, stride=[1, 1, 1, 1])
                + self.b_conv3,
                scale=True,
                center=True,
                training=train_mode,
            )
            self.h_conv3 = tf.nn.leaky_relu(self.norm3, alpha=0.1)
        if verbose:
            print(self.h_conv3.get_shape())

        self.final_layer = self.fc_layer(
            self.h_conv3,
            self.get_size(self.h_conv3),
            np.prod(output_shape),
            "final_layer",
        )
        self.final_layer = tf.nn.softmax(self.final_layer)
        self.output = tf.identity(self.final_layer, name="output")
        if verbose:
            print(self.output.get_shape())

    def conv2d(self, x, W, stride=[1, 1, 1, 1]):
        """conv2d returns a 2d convolution layer with full stride."""
        # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.conv2d(x, W, strides=stride, padding="SAME")

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    def max_pool_2x2_1(self, x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME"
        )

    def weight_variable(self, shape, var_name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name=var_name, initializer=initial)

    def bias_variable(self, shape, var_name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.001, shape=shape)
        return tf.get_variable(name=var_name, initializer=initial)

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name, init_type="xavier"):
        if init_type == "xavier":
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False),
            ]
        else:
            weight_init = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], 0.0, 0.001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolean to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1]
                )
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            # var = tf.get_variable(name=var_name, initializer=value)

        self.var_dict[(name, idx)] = var

        return var
