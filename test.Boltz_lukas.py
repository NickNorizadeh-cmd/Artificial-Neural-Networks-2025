import numpy as np
from itertools import product
import math
# Energy function
def H(W, h_vec, v_vec, Theta_v, Theta_h):   
    h_vec = np.array(h_vec)
    v_vec = np.array(v_vec)
    energy = -(h_vec.T @ W @ v_vec) + Theta_v.T @ v_vec + Theta_h.T @ h_vec
    return energy.item()

# Boltzmann joint probability
def boltzmann_dist(W, h_vec, v_vec, Theta_v, Theta_h):
    M = W.shape[0]
    N = W.shape[1]
    visible_patterns = list(product([-1, 1], repeat=N))
    hidden_patterns = list(product([-1, 1], repeat=M))
    Z = 0
    for v in visible_patterns:
        for h in hidden_patterns:
            Z += math.exp(-H(W, h, v, Theta_v, Theta_h))
    return math.exp(-H(W, h_vec, v_vec, Theta_v, Theta_h)) / Z


h_test = np.array([[1],[1]])
W_test = np.array([[1,0],[0,1]])
theta_test = np.array([[0],[0]])
v_test = np.array([[1],[1]])
print(H(W_test, h_test, v_test, theta_test, theta_test))
print(boltzmann_dist(W_test, h_test, v_test, theta_test, theta_test))