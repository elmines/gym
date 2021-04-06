# 3rd Party
import gym
import tensorflow as tf
# Local
from . import NUM_ACTIONS

def mlp(input_shape) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(NUM_ACTIONS),
        tf.keras.layers.Softmax()
    ])
    return model