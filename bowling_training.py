# Python STL
import pdb
# 3rd Party
import gym
import numpy as np
import tensorflow as tf
from PIL import Image
# Local
from bowling_model import preprocess, to_grayscale

env = gym.make("Bowling-v0")
pdb.set_trace()
observation = env.reset()
observation = preprocess(observation)

ACTION_NAMES = ["NOOP", "FIRE", "UP", "DOWN"]
ACTIONS = [0, 1, 2, 3]
NUM_ACTIONS = 3
# Hyperparameters
batch_size = 10
learning_rate = 1e-3

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=observation.shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(NUM_ACTIONS),
    tf.keras.layers.Softmax()
])
