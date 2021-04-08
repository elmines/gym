# 3rd Party
import gym
import tensorflow as tf
# Local
from . import NUM_ACTIONS

def mlp(input_shape, seed : int = 0) -> tf.keras.Model:
    cur_seed = seed
    layers = []
    layers.append(tf.keras.layers.InputLayer(input_shape=input_shape))
    layers.append(tf.keras.layers.Flatten())
    layers.append(
        tf.keras.layers.Dense(200,
                            bias_initializer="zeros",
                            kernel_initializer=tf.keras.initializers.GlorotUniform(cur_seed)
        )
    )
    seed += 1
    layers.append(tf.keras.layers.ReLU())
    layers.append(
        tf.keras.layers.Dense(NUM_ACTIONS,
                            bias_initializer="zeros",
                            kernel_initializer=tf.keras.initializers.GlorotUniform(cur_seed))
    )
    seed += 1
    layers.append(tf.keras.layers.Softmax())

    model = tf.keras.Sequential(layers)
    return model