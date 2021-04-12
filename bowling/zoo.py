# Python STL
from typing import List
# 3rd Party
import gym
import tensorflow as tf
# Local
from . import NUM_ACTIONS

class mlp(tf.keras.Model):

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

__all__ = ["mlp"]