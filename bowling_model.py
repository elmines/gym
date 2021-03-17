# 3rd Party
import numpy as np
from PIL import Image

def preprocess(observation: np.ndarray) -> np.ndarray:
    observation = trim(observation)
    observation = to_grayscale(observation)
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