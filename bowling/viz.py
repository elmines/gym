from PIL import Image
import numpy as np
def display(observation: np.ndarray):
    if observation.shape[-1] != 3:
        observation = np.stack([observation] * 3, axis=-1)
    Image.fromarray(observation).show()
