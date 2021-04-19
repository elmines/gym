# Python STL
import pdb
import time
from typing import List, Dict, Union, Tuple
import os
import json
# 3rd Party
import gym
import numpy as np
import tensorflow as tf

from PIL import Image
# Local
from bowling import preprocess, to_grayscale, make_grad_buffer, discount_rewards
from bowling import make_random_baseline, make_uniform_noise_func, select_schedule_item
from bowling import make_model_wrapper, evaluate
from bowling import ACTION_NAMES, ACTION_DICT, NUM_ACTIONS, NOOP

def train(
    model                 : tf.keras.Model,
    env,
    preproc_func,
    weights_load          : str                           = None,                         # Weight file management
    save_dir              : str                           = None,
    save_freq             : int                           = None,
    gamma                 : float                         = 0.25,                         # MDP Parameters
    choice_freq           : int                           = 5,
    optimizer             : tf.keras.optimizers.Optimizer = None,                         # ML Parameters
    clip_range            : Tuple[float,float]            = None,                       
    max_batches           : int                           = 10,
    batch_size            : int                           = 1,
    schedule_thresholds   : List[float]                   = [0., 30., 50., 100.],         # Schedule Parameters
    max_choices_schedule  : List[int]                     = 600,         
    noise_schedule        : List[float]                   = 0,
    render                : bool                          = False,                        # Misc. Parameters
    render_eval           : bool                          = False,
    verbose               : bool                          = False,
    num_eval_samples      : int                           = 1,
    seed                  : int                           = 0,
    epsilon               : float                         = 0.001,
    noise_func                                            = None):

    if not save_dir:
        save_dir = os.path.join("weights", str(int(time.time())))
        os.makedirs(save_dir, exist_ok=True)

    results = {
        "save_dir"            : save_dir,
        "gamma"               : gamma,
        "schedule_thresholds" : schedule_thresholds,
        "max_choices"         : max_choices_schedule,
        "choice_freq"         : choice_freq,
        "optimizer"           : str(optimizer),
        "clip_range"          : list(clip_range) if clip_range else None,
        "max_batches"         : str(max_batches),
        "schedule_thresholds" : schedule_thresholds,
        "seed"                : seed,
        "best_eval_mean"      : -1,
        "train_scores"        : [],
        "eval_scores"         : [],
    }


    if clip_range:
        clip_func = lambda t: tf.clip_by_value(t, *clip_range)
    else:
        clip_func = lambda t: t

    if not optimizer: optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    action_rng           = np.random.default_rng(seed)
    model_wrapper        = make_model_wrapper(model, preproc_func, action_rng)

    ################ Initialize Schedule Parameters #################################
    schedule_thresholds  = np.squeeze(np.array(schedule_thresholds, dtype=np.float32))
    def expand_if_scalar(schedule):
        schedule = np.squeeze(np.array(schedule, dtype=np.float32))
        if np.ndim(schedule) == 0: return np.repeat(schedule, len(schedule_thresholds))
        return schedule
    max_choices_schedule = expand_if_scalar(max_choices_schedule)
    noise_schedule       = expand_if_scalar(noise_schedule)

    noise_weight         = select_schedule_item(0, noise_schedule, schedule_thresholds)
    max_choices          = select_schedule_item(0, max_choices_schedule, schedule_thresholds)

    if not noise_func:
        noise_func = make_uniform_noise_func(NUM_ACTIONS)

    ################## Training Loop ######################
    advantages                                     = []
    grad_buffer   : List[List[tf.Tensor]]          = [list() for v in model.trainable_variables]
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
            #print(raw_gradient)
            # Take actual action
            action = ACTION_DICT[action_index]
            obs, step_reward, done, _ = env.step(action)
            recent_reward += step_reward
            last_action = action
            # Track history for later network updates
            obs_history.append(preproc_obs)
            rew_history.append(recent_reward)
            for (sub_list, grad) in zip(grad_buffer, raw_gradient):
                sub_list.append(grad)
            choices_made += 1
            recent_reward = 0
        else:
            obs, step_reward, done, _    = env.step(last_action)
            recent_reward += step_reward

        # Terminate episode
        if done or (choices_made >= max_choices):
            train_score = sum(rew_history) + recent_reward
            advantages.extend(discount_rewards(rew_history, gamma=gamma))
            rew_history = []
            obs_history = []

            # Update model parameters
            if ep_number % batch_size == 0:
                num_samples = len(grad_buffer[0])
                advantages  = np.array(advantages)
                grad_buffer = [tf.stack(sub_list) for sub_list in grad_buffer]
                for i in range(len(grad_buffer)):
                    reshaped_adv   = np.reshape(advantages, [len(advantages)] + [1]*(len(grad_buffer[i].shape)-1))
                    grad_buffer[i] = tf.math.reduce_sum(reshaped_adv * grad_buffer[i], axis=0)
                    grad_buffer[i] *= -1 # TF Optimizers implicitly multiply the gradients by -1, so we need to undo this
                optimizer.apply_gradients( zip(grad_buffer, model.trainable_variables) )
                advantages = []
                grad_buffer = [list() for v in model.trainable_variables]

            eval_scores = [evaluate(env, model_wrapper, choice_freq, render=render_eval) for _ in range(num_eval_samples)]
            avg_eval_score = sum(eval_scores) / num_eval_samples 

            # Update schedule variables
            noise_weight = select_schedule_item(avg_eval_score, noise_schedule, schedule_thresholds)
            max_choices  = select_schedule_item(avg_eval_score, max_choices_schedule, schedule_thresholds)
            if verbose:
                print(f"Episode {ep_number} completed!")
                print(f"train_score={train_score}, eval_scores={eval_scores}")

            # Save Latest
            if save_freq and (ep_number % save_freq) == 0:
                model.save_weights(os.path.join(save_dir, "latest.h5"))
            if avg_eval_score > best_score:
                model.save_weights(os.path.join(save_dir, "best.h5"))
                best_score = avg_eval_score
                results["best_eval_mean"] = { str(ep_number) : best_score }
            results["eval_scores"].append(eval_scores)

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

    return results
