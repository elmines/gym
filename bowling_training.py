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
from bowling_model import discount_rewards, make_random_baseline, make_uniform_noise_func

np.random.seed(0)

env = gym.make("Bowling-v0")
observation = env.reset()
observation = preprocess(observation)

render = True
ACTION_NAMES = ["NOOP", "FIRE", "UP", "DOWN"]
ACTION_DICT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}
NUM_ACTIONS = len(ACTION_NAMES)
# Hyperparameters
### Supervised Learning Params
max_batches     = 10
batch_size      = 1
learning_rate   = 1e-3
### RL Params
gamma           = 0.99
max_steps       = 500
### Scheduled Hyperparameters
thresholds           = np.array([0., 1., 2., 3., 50., 100.])
max_steps_schedule   = np.array([500, 500, 500, 500, 500, 500], dtype=np.int32)
noise_schedule       = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
noise_func           = make_uniform_noise_func(NUM_ACTIONS)
def select_schedule_item(score, schedule, thresholds):
    assert len(schedule) == len(thresholds)
    return schedule[ np.max( (score >= thresholds)*np.arange(len(thresholds))) ]
noise_weight  = select_schedule_item(0, noise_schedule, thresholds)
max_steps     = select_schedule_item(0, max_steps_schedule, thresholds)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=observation.shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(NUM_ACTIONS),
    tf.keras.layers.Softmax()
])


################## Training Loop ######################
advantages                                     = []
grad_buffer   : List[tf.Tensor]                = []
obs_history   : List[object]                   = []
rew_history   : List[float]                    = []
ep_number     : int                            = 1
t             : int                            = 1
obs                                            = env.reset()
finished_training                              = False
while not finished_training:
    if render: env.render()
    preproc_obs = preprocess(obs)
    # Compute gradients for updating later
    with tf.GradientTape() as tape:
        aprobs       = tf.squeeze(model(tf.expand_dims(preproc_obs, axis=0)))
        aprobs = (1-noise_weight)*aprobs + noise_weight*noise_func()
        try:
            action_index = np.random.choice(range(NUM_ACTIONS), p=aprobs)
        except ValueError as err:
            if str(err) != "probabilities do not sum to 1": raise err
            action_index = np.argmax(aprobs)
        action_prob  = aprobs[action_index]
        raw_gradient = tape.gradient(tf.math.log(action_prob), model.trainable_variables)
    # Take actual action
    action = ACTION_DICT[action_index]
    obs, reward, done, _ = env.step(action)
    # Track history for later network updates
    obs_history.append(preproc_obs)
    rew_history.append(reward)
    grad_buffer.append(raw_gradient)
    # Terminate episode
    if done or (t >= max_steps):
        bowling_score = sum(rew_history)
        advantages.extend(discount_rewards(rew_history, gamma=gamma))
        rew_history = []
        obs_history = []
        # Update model parameters
        if ep_number % batch_size == 0:
            for (adv, grad) in zip(np.array(advantages), grad_buffer):
                for (v, v_inc) in zip(model.trainable_variables, grad):
                    v.assign_add(adv*v_inc)
            advantages = []
            grad_buffer = []

        print(f"Episode {ep_number} finished. Score = {bowling_score}")
        # Update schedule variables
        noise     = select_schedule_item(bowling_score, noise_schedule, thresholds)
        max_steps = select_schedule_item(bowling_score, max_steps_schedule, thresholds)

        # Update episodal variables
        t              = 1
        ep_number     += 1
        obs            = env.reset()
        if ep_number > batch_size*max_batches:
            finished_training = True
        else:
            pass
    else:
        t += 1