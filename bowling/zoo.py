# Python STL
from typing import List
# 3rd Party
import gym
import tensorflow as tf
import numpy as np
# Local
from .preprocess import flatten, to_grayscale, trim
from . import NUM_ACTIONS

class mlp(tf.keras.Model):

    @staticmethod
    def preprocess(x):
        return flatten(to_grayscale(trim(x)))

    def __init__(self, input_shape, seed : int = 0):
        super(mlp, self).__init__()
        cur_seed = seed

        self.flatten = tf.keras.layers.Flatten()

        self.dense1  = tf.keras.layers.Dense(200,
                                bias_initializer="zeros",
                                kernel_initializer=tf.keras.initializers.GlorotUniform(cur_seed)
            )
        seed += 1

        self.nonlin = tf.keras.layers.Activation(tf.keras.activations.tanh)

        self.dense2 = tf.keras.layers.Dense(NUM_ACTIONS,
                                bias_initializer="zeros",
                                kernel_initializer=tf.keras.initializers.GlorotUniform(cur_seed)
        )
        seed += 1

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x     = self.flatten(inputs)
        h1    = self.dense1(x)
        act1  = self.nonlin(h1)
        h2    = self.dense2(act1)
        probs = self.softmax(h2)
        return probs

class ConvNet(tf.keras.Model):

    @staticmethod
    def preprocess(x):
        x = trim(x)
        #x = to_grayscale(x)
        #x = tf.expand_dims(x, axis=-1) 
        x = tf.cast(x, tf.float32)
        return x

    def __init__(self, input_shape, seed : int = 0):
        super(ConvNet, self).__init__()
        cur_seed : int = seed

        self.conv = tf.keras.layers.Conv2D(8, [3,3],
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=cur_seed))
        cur_seed += 1

        self.nonlin = tf.keras.layers.Activation(tf.keras.activations.tanh)

        self.pool = tf.keras.layers.MaxPool2D([2,2], strides=[2,2], padding="valid")

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(NUM_ACTIONS,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=cur_seed))
        cur_seed += 1

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x      = self.conv(inputs)
        x      = self.nonlin(x)
        x      = self.pool(x)
        x      = self.flatten(x)
        logits = self.dense(x)
        probs  = self.softmax(logits)
        return probs

__all__ = ["mlp", "ConvNet"]