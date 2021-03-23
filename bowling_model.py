# 3rd Party
import numpy as np
from PIL import Image

def discount_rewards(rewards, gamma) -> np.ndarray:
    G = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        G.append(rewards[i] + gamma*rewards[i+1] )
    G = list(reversed(G))
    G = np.array(G)
    return G

def make_grad_buffer(*variables):
    return [np.zeros(v.shape) for v in variables]

def preprocess(observation: np.ndarray) -> np.ndarray:
    observation = trim(observation)
    observation = to_grayscale(observation)
    observation = np.squeeze(observation.flatten())
    return observation

def trim(observation: np.ndarray) -> np.ndarray:
    return observation[80:-36]

def to_grayscale(observation: np.ndarray) -> np.ndarray:
    orig_type = observation.dtype
    observation = np.sum(observation, axis=-1) / 3
    observation = observation.astype(orig_type)
    return observation

def display(observation: np.ndarray):
    if observation.shape[-1] != 3:
        observation = np.stack([observation] * 3, axis=-1)
    Image.fromarray(observation).show()

__all__ = ["preprocess", "trim", "to_grayscale", "display"]