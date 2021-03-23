# Python STL
import pdb
from typing import List, Dict, Union
# 3rd Party
import gym
import numpy as np
import tensorflow as tf
from PIL import Image
# Local
from bowling_model import preprocess, to_grayscale, make_grad_buffer, discount_rewards

env = gym.make("Bowling-v0")
observation = env.reset()
observation = preprocess(observation)

render = False
ACTION_NAMES = ["NOOP", "FIRE", "UP", "DOWN"]
ACTION_DICT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}
NUM_ACTIONS = len(ACTION_NAMES)
# Hyperparameters
batch_size = 10
learning_rate = 1e-3
gamma = 0.99
baseline_func = lambda s: 0

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=observation.shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(NUM_ACTIONS),
    tf.keras.layers.Softmax()
])


################## Training Loop ######################
grad_buffer                                    = []
ep_baselines  : List[float]                    = []
ep_rewards    : List[float]                    = []
advantages                                     = []
ep_number                                      = 1
obs                                            = env.reset()
max_episodes                                   = 2
finished_training                              = False
while not finished_training:
    if render: env.render()
    preproc_obs = tf.constant(preprocess(obs))
    preproc_obs = tf.expand_dims(preproc_obs, axis=0) # Keras requires a batch dimension
    # Compute gradients for updating later
    with tf.GradientTape() as tape:
        aprobs       = tf.squeeze(model(preproc_obs))
        action_index = np.random.choice(range(NUM_ACTIONS), p=aprobs)
        action_prob  = aprobs[action_index]
        raw_gradient = tape.gradient(tf.math.log(action_prob), model.trainable_variables)
    # Take actual action
    old_obs = obs
    action = ACTION_DICT[action_index]
    obs, reward, done, _ = env.step(action)
    # Track history for later network updates
    ep_baselines.append(baseline_func(old_obs))
    ep_rewards.append(reward)
    grad_buffer.append(raw_gradient)
    # Perform update
    if done:
        drewards      = discount_rewards(ep_rewards, gamma)
        ep_advantages = drewards - np.array(ep_baselines)
        advantages.extend(ep_advantages)

        # Update model parameters
        if ep_number % batch_size == 0:
            advantages = np.array(advantages)
            for (adv, grad) in zip(advantages, grad_buffer):
                for (v, v_inc) in zip(model.trainable_variables, grad):
                    v.assign_add(adv*grad)
            # Update batch-wise variables
            grad_buffer = []
            advantages  = []
        
        # Update episodal variables
        ep_number     += 1
        ep_baselines   = []
        if ep_number > max_episodes:
            finished_training = True