# Python STL
import pdb
from typing import List, Dict, Union
# 3rd Party
import gym
import numpy as np
import tensorflow as tf
from PIL import Image
# Local
from bowling_model import preprocess, to_grayscale, make_grad_buffer
from bowling_model import discount_rewards, make_random_baseline

np.random.seed(0)

env = gym.make("Bowling-v0")
observation = env.reset()
observation = preprocess(observation)

render = False
MAX_STEPS = 200
ACTION_NAMES = ["NOOP", "FIRE", "UP", "DOWN"]
ACTION_DICT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}
NUM_ACTIONS = len(ACTION_NAMES)
# Hyperparameters
max_batches     = 2
batch_size      = 2
learning_rate   = 1e-3
gamma           = 0.99
baseline_func   = make_random_baseline(seed = 0)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=observation.shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(NUM_ACTIONS),
    tf.keras.layers.Softmax()
])


################## Training Loop ######################
advantages = []
grad_buffer   : List[tf.Tensor]                = []
obs_history   : List[object]                   = []
rew_history   : List[float]                    = []
ep_number     : int                            = 1
t             : int                            = 1
obs                                            = env.reset()
finished_training                              = False
print(f"Episode {ep_number}")
while not finished_training:
    if render: env.render()
    preproc_obs = preprocess(obs)
    # Compute gradients for updating later
    with tf.GradientTape() as tape:
        aprobs       = tf.squeeze(model(tf.expand_dims(preproc_obs, axis=0)))
        action_index = np.random.choice(range(NUM_ACTIONS), p=aprobs)
        action_prob  = aprobs[action_index]
        raw_gradient = tape.gradient(tf.math.log(action_prob), model.trainable_variables)
    # Take actual action
    action = ACTION_DICT[action_index]
    obs, reward, done, _ = env.step(action)
    # Track history for later network updates
    obs_history.append(preproc_obs)
    rew_history.append(reward)
    grad_buffer.append(raw_gradient)
    # Perform update
    if done or (t >= MAX_STEPS):
        advantages.extend(discount_rewards(rew_history, gamma=gamma) - baseline_func(obs_history))
        rew_history = []
        obs_history = []
        # Update model parameters
        if ep_number % batch_size == 0:
            for (adv, grad) in zip( np.array(advantages), grad_buffer):
                for (v, v_inc) in zip(model.trainable_variables, grad):
                    v.assign_add(adv*v_inc)
        # Update episodal variables
        t              = 1
        ep_number     += 1
        if ep_number > batch_size*max_batches:
            finished_training = True
        else:
            print(f"Episode {ep_number}")
    else:
        t += 1