import numpy as np

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
