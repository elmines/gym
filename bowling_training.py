# Python STL
import pdb
import time
from typing import List, Dict, Union
import os
# 3rd Party
import gym
import numpy as np
import tensorflow as tf

from PIL import Image
# Local
from bowling import preprocess, to_grayscale, make_grad_buffer, discount_rewards
from bowling import make_random_baseline, make_uniform_noise_func, select_schedule_item
from bowling import ACTION_NAMES, ACTION_DICT, NUM_ACTIONS

def train(
    model                 : tf.keras.Model,
    preproc_func,
    weights_load          : str             = None,
    save_dir              : str             = None,
    save_freq             : int             = None,
    max_batches           : int             = 10,
    batch_size            : int             = 1,
    lr                    : float           = 1e-2,
    gamma                 : float           = 0.99,
    render                : bool            = False,
    seed                  : int             = 0,
    schedule_thresholds   : List[float]     = [0.,    10., 15., 20.],
    lr_schedule           : List[float]     = [0.001, 0.001, 0.001, 0.001],
    max_steps_schedule    : List[int]       = [600, 600, 600, 600],
    noise_schedule        : List[float]     = [0.75, 0.5, 0.25, 0.1],
    epsilon               : float           = 0.001,
    noise_func                           = None):

    if not save_dir:
        save_dir = os.path.join("weights", str(int(time.time())))
        os.makedirs(save_dir, exist_ok=True)

    schedule_thresholds   = np.array(schedule_thresholds, dtype=np.float32)
    max_steps_schedule = np.array(max_steps_schedule, dtype=np.float32)
    noise_schedule     = np.array(noise_schedule, dtype=np.float32)

    noise_weight       = select_schedule_item(0, noise_schedule, schedule_thresholds)
    max_steps          = select_schedule_item(0, max_steps_schedule, schedule_thresholds)
    lr                 = select_schedule_item(0, lr_schedule, schedule_thresholds)

    if not noise_func:
        noise_func = make_uniform_noise_func(NUM_ACTIONS)
    action_rng         = np.random.default_rng(seed)

    env = gym.make("Bowling-v0")
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
        preproc_obs = preproc_func(obs)
        preproc_obs = tf.constant(preproc_obs)
        preproc_obs = tf.expand_dims(preproc_obs, axis=0)
        # Compute gradients for updating later
        with tf.GradientTape() as tape:
            raw_aprobs = tf.squeeze(model(preproc_obs))
            aprobs     = (1-noise_weight)*raw_aprobs + noise_weight*noise_func()
            try:
                action_index = action_rng.choice(range(NUM_ACTIONS), p=aprobs)
            except ValueError as err:
                if str(err) != "probabilities do not sum to 1": raise err
                action_index = action_rng.integers(0, NUM_ACTIONS)
            raw_action_prob  = raw_aprobs[action_index]
            log_prob         = tf.math.log(raw_action_prob)
        raw_gradient = tape.gradient(log_prob, model.trainable_variables)
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
                        v.assign_add(lr*adv*v_inc)
                advantages = []
                grad_buffer = []

            # Update schedule variables
            noise_weight = select_schedule_item(bowling_score, noise_schedule, schedule_thresholds)
            max_steps    = select_schedule_item(bowling_score, max_steps_schedule, schedule_thresholds)
            lr           = select_schedule_item(bowling_score, lr_schedule, schedule_thresholds)
            print(f"Episode {ep_number} finished. Score = {bowling_score}")
            print(f"noise_weight={noise_weight}, max_steps={max_steps}, lr={lr}")

            if save_freq and (ep_number % save_freq) == 0:
                model.save_weights(os.path.join(save_dir, "latest.h5"))


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