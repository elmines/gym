import numpy as np

def discount_rewards(rewards, gamma) -> np.ndarray:
    G = [rewards[-1]]
    for i in range(len(rewards) - 2, -1, -1):
        G.append(rewards[i] + gamma*rewards[i+1] )
    G = list(reversed(G))
    G = np.array(G)
    return G

def make_grad_buffer(*variables):
    return [np.zeros(v.shape) for v in variables]

