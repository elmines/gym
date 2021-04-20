import json
import pdb
from itertools import product
import os

import gym
import tensorflow as tf
from train import train
from bowling import to_grayscale, trim, flatten, StackStateWrapper

if __name__ == "__main__":
    from bowling import zoo

    #gammas       = [0.25, 0.5, 0.75, 0.99, 1.]
    gammas       = [0.99]
    stack_frames = [1, 2, 4]
    choice_freqs = [1, 2, 4]
    model_class = zoo.ConvNet
    lrs = [0.0001]
    optimizers = [lambda lr: tf.keras.optimizers.Adam(lr), lambda lr: tf.keras.optimizers.SGD(lr)] 

    i = 0
    for (gamma, stack_frame, choice_freq, lr, opt_lambda) in product(gammas, stack_frames, choice_freqs, lrs, optimizers):
        env = gym.make("Bowling-v0")
        if stack_frame > 1:
            env = StackStateWrapper(env, stack_frame)

        preproc_func = model_class.preprocess
        preproc_obs  = preproc_func(env.reset())
        input_shape  = preproc_obs.shape
        model        = model_class(input_shape)
        model(tf.expand_dims(preproc_obs, axis=0)) # Force build

        save_dir = os.path.join("weights", str(i))
        os.makedirs(save_dir, exist_ok=True)
        results = train(model, env, preproc_func,
                gamma                = gamma,
                save_dir             = save_dir,
                render               = False,
                render_eval          = False,
                choice_freq          = choice_freq,
                save_freq            = 1,
                max_batches          = 50,
                optimizer            = opt_lambda(lr),
                num_eval_samples     = 5,
                schedule_thresholds  = [0.,      30.,    50., 100.],
                noise_schedule       = 0,
                max_choices_schedule = 600,
            )

        results["lr"] = lr
        results["model"] = "Conv"
        results["stack_frame"] = stack_frame
        with open( os.path.join(save_dir, "results.json"), "w") as w:
            w.write( json.dumps(results, indent=2) )
        i += 1
