# STL
import json
import pdb
from itertools import product
import os
import numpy as np
# 3rd Party
import gym
import tensorflow as tf
#Local
from bowling import to_grayscale, trim, flatten, StackStateWrapper, make_model_wrapper, evaluate
from bowling import zoo


def evaluate_dir(model_dir, num_games):
    model_class = zoo.ConvNet
    with open(os.path.join(model_dir, "results.json"), "r") as r:
        results_obj = json.load(r)
    stack_frame = results_obj["stack_frame"]
    choice_freq = results_obj["choice_freq"]

    env = gym.make("Bowling-v0")
    if stack_frame > 1:
        env = StackStateWrapper(env, stack_frame)
    preproc_func = model_class.preprocess
    preproc_obs  = preproc_func(env.reset())
    input_shape  = preproc_obs.shape
    model        = model_class(input_shape)
    model(tf.expand_dims(preproc_obs, axis=0)) # Force build

    action_rng = np.random.default_rng(0)
    f = make_model_wrapper(model, preproc_func, action_rng)

    model.load_weights( os.path.join(model_dir, "latest.h5") )
    fully_trained_scores = [evaluate(env, f, choice_freq, render=False) for _ in range(num_games)]

    model.load_weights( os.path.join(model_dir, "best.h5") )
    cherry_picked_scores = [evaluate(env, f, choice_freq, render=False) for _ in range(num_games)]

    with open( os.path.join(model_dir, "scores.json"), "w") as w:
        w.write(json.dumps({
            "fully_trained": {
                "scores" : fully_trained_scores,
                "avg"    : sum(fully_trained_scores)/len(fully_trained_scores)
            },
            "cherry_picked": {
                "scores" : cherry_picked_scores,
                "avg"    : sum(cherry_picked_scores)/len(cherry_picked_scores)
            }
        }, indent=2) + "\n")

if __name__ == "__main__":
    num_games = 20
    for results_root in [ os.path.join(os.environ['HOME'],"resultsA"), os.path.join(os.environ["HOME"], "resultsB") ]:
        for sub_dir  in os.listdir(results_root):
            model_dir = os.path.join(results_root, sub_dir)
            evaluate_dir(model_dir, num_games)
