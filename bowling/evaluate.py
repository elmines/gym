# Local
from . import ACTION_DICT
# 3rd Party
import tensorflow as tf
import numpy as np

def evaluate(env, f, choice_freq, render=False) -> float:
    obs = env.reset()
    tot_reward = 0
    done = False
    t = 0
    while not done:
        if render: env.render()
        if t % choice_freq == 0: action = f(obs)
        obs, reward, done, _ = env.step(action)
        tot_reward += reward
        t += 1
    return tot_reward

def make_model_wrapper(model, preproc_func, rng):
    def f(obs):
        preproc_obs = preproc_func(obs)
        preproc_obs = tf.constant(preproc_obs)
        preproc_obs = tf.expand_dims(preproc_obs, axis=0)
        aprobs = tf.squeeze(model(preproc_obs))
        actions = list(range(len(aprobs)))
        try:
            action_index = rng.choice(actions, p=aprobs)
        except ValueError as err:
            if str(err) != "probabilities do not sum to 1": raise err
            if rng.uniform() <= 0.9:
                action_index = np.argmax(aprobs)
            else:
                action_index = rng.choice(actions)
        return ACTION_DICT[action_index]
    return f
