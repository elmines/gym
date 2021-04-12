# Python STL
import pdb
import time
from typing import List, Dict, Union, Tuple
import os
# 3rd Party
import gym
import numpy as np
import tensorflow as tf

from PIL import Image
# Local
from bowling import preprocess, to_grayscale, make_grad_buffer, discount_rewards
from bowling import make_random_baseline, make_uniform_noise_func, select_schedule_item
from bowling import ACTION_NAMES, ACTION_DICT, NUM_ACTIONS, NOOP

def train(
    model                 : tf.keras.Model,
    preproc_func,
    choice_freq           : int                = 1,
    weights_load          : str                = None,
    save_dir              : str                = None,
    save_freq             : int                = None,
    clip_range            : Tuple[float,float] = (-2,2),
    max_batches           : int                = 10,
    batch_size            : int                = 1,
    gamma                 : float              = 0.99,
    render                : bool               = False,
    seed                  : int                = 0,
    schedule_thresholds   : List[float]        = [0.,    10., 15., 20.],
    lr_schedule           : List[float]        = [0.001, 0.001, 0.001, 0.001],
    max_actions_schedule  : List[int]          = [600, 600, 600, 600],
    noise_schedule        : List[float]        = [0.75, 0.5, 0.25, 0.1],
    epsilon               : float              = 0.001,
    noise_func                                 = None):

    if not save_dir:
        save_dir = os.path.join("weights", str(int(time.time())))
        os.makedirs(save_dir, exist_ok=True)

    if clip_range:
        clip_func = lambda t: tf.clip_by_value(t, *clip_range)
    else:
        clip_func = lambda t: t

    schedule_thresholds   = np.array(schedule_thresholds, dtype=np.float32)
    max_actions_schedule = np.array(max_actions_schedule, dtype=np.float32)
    noise_schedule     = np.array(noise_schedule, dtype=np.float32)

    noise_weight       = select_schedule_item(0, noise_schedule, schedule_thresholds)
    max_actions        = select_schedule_item(0, max_actions_schedule, schedule_thresholds)
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
    choices_made  : int                            = 0
    last_action   : int                            = NOOP
    recent_reward : int                            = 0
    best_score    : int                            = -1

    obs                                            = env.reset()
    finished_training                              = False
    while not finished_training:
        if render: env.render()

        if t % choice_freq == 0:
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
                log_prob         = tf.math.log(raw_action_prob + epsilon)
            raw_gradient = tape.gradient(log_prob, model.trainable_variables)
            raw_gradient = list(map(clip_func, raw_gradient))
            # Take actual action
            action = ACTION_DICT[action_index]
            obs, step_reward, done, _ = env.step(action)
            recent_reward += step_reward
            last_action = action
            # Track history for later network updates
            obs_history.append(preproc_obs)
            rew_history.append(recent_reward)
            grad_buffer.append(raw_gradient)
            choices_made += 1
            recent_reward = 0
        else:
            obs, step_reward, done, _    = env.step(last_action)
            recent_reward += step_reward

        # Terminate episode
        if done or (choices_made >= max_actions):
            bowling_score = sum(rew_history) + recent_reward
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
            max_actions  = select_schedule_item(bowling_score, max_actions_schedule, schedule_thresholds)
            lr           = select_schedule_item(bowling_score, lr_schedule, schedule_thresholds)
            print(f"Episode {ep_number} finished. Score = {bowling_score}")
            print(f"noise_weight={noise_weight}, max_actions={max_actions}, lr={lr}")

            # Save Latest
            if save_freq and (ep_number % save_freq) == 0:
                model.save_weights(os.path.join(save_dir, "latest.h5"))
            if bowling_score > best_score:
                print("New best score!")
                model.save_weights(os.path.join(save_dir, "best.h5"))
                best_score = bowling_score

            # Update episodal variables
            t              = 1
            choices_made   = 0
            ep_number     += 1
            obs            = env.reset()
            if ep_number > batch_size*max_batches:
                finished_training = True
            else:
                pass
        else:
            t += 1
