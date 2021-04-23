from bowling import NUM_ACTIONS, evaluate
import gym
import numpy as np
import json


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    num_games = 20
    f = lambda s: rng.choice(NUM_ACTIONS)
    env = gym.make("Bowling-v0")
    scores = [evaluate(env, f, 1, render=False) for _ in range(num_games)]

    with open("random_scores.json", "w") as w:
        w.write(
            json.dumps({
                "scores": scores,
                "avg": sum(scores)/len(scores)
            }, indent=2) + "\n"
        )
