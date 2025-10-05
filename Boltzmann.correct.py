import numpy as np
import random
import math
from itertools import product
import matplotlib.pyplot as plt

# XOR dataset: 4 patterns with equal probability
xor_data = np.array([
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
    [-1, -1, -1]
])
hidden_units = [1, 2, 4, 8]
M = hidden_units[2]  # You can loop over this later
v_max = 10000
p0 = 20
k = 10
eta = 0.005
N = 3  # Number of visible units

def p(b_i):
    return 1 /(1 + np.exp(-2 * b_i))

def kl_divergence(true_dist, model_dist):
    kl = 0.0
    for pattern in true_dist:
        p_val = true_dist[pattern]
        q_val = model_dist.get(pattern, 1e-12)
        kl += p_val * np.log(p_val / q_val)
    return kl

def theoretical_bound(M, N=3):
    threshold = 2**(N - 1) - 1
    if M >= threshold:
        return 0.0
    log_term = np.floor(np.log2(M + 1))
    bound = np.log(2) * (N - log_term - (M + 1) / (2**log_term))
    return bound

# True XOR distribution
true_dist = {tuple(p): 0.25 for p in xor_data}

# Initialize weights and biases
print(f"\nTraining RBM with {M} hidden units...")
W = np.random.randn(M, N)
Theta_h = np.zeros((M, 1))
Theta_v = np.zeros((N, 1))

# CD-k training loop
for epoch in range(v_max):
    x_sample = np.random.choice(range(xor_data.shape[0]), size=p0)
    delta_w_m_n = np.zeros(W.shape)
    delta_theta_v = np.zeros((N, 1))
    delta_theta_h = np.zeros((M, 1))

    for mu in range(p0):
        pattern = xor_data[x_sample[mu]].copy() # the idea is to copy the xor_data and let v_vec point at the copy. 
        v_vec = pattern # Instead of having v_vec reference xor_data so that each time v_vec changes so does xor_data. Now it just changes v_vec each iteration
        v_test = xor_data[x_sample[mu]].reshape((N, 1)) 
        print("v_test is", v_test)

        b_h = W @ v_vec - Theta_h
        h_vec = np.zeros((M, 1))

        for i in range(M):
            r = random.random()
            prob = p(b_h[i])
            h_vec[i] = 1 if r < prob else -1

        b_h_0 = b_h.copy()
        v_vec_0 = v_vec.copy()

        for t in range(k):
            b_v = W.T @ h_vec - Theta_v
            for j in range(N):
                r = random.random()
                prob = p(b_v[j])
                v_vec[j] = 1 if r < prob else -1

            b_h = W @ v_vec - Theta_h
            for i in range(M):
                r = random.random()
                prob = p(b_h[i])
                h_vec[i] = 1 if r < prob else -1

        first_term = np.outer(np.tanh(b_h_0), v_vec_0.T)
        second_term = np.outer(np.tanh(b_h), v_vec.T)
        delta_w_m_n += eta * (first_term - second_term)
        delta_theta_v += eta * (v_vec_0 - v_vec)
        delta_theta_h += eta * (np.tanh(b_h_0) - np.tanh(b_h))

    W += delta_w_m_n
    Theta_h += delta_theta_h
    Theta_v += delta_theta_v

# Energy function
def H(W, h_vec, v_vec, Theta_v, Theta_h):   
    h_vec = np.array(h_vec).reshape((W.shape[0], 1))
    v_vec = np.array(v_vec).reshape((W.shape[1], 1))
    energy = -(h_vec.T @ W @ v_vec) + Theta_v.T @ v_vec + Theta_h.T @ h_vec
    return energy.item()

# Boltzmann joint probability
def boltzmann_dist(W, h_vec, v_vec, Theta_v, Theta_h):
    visible_patterns = list(product([-1, 1], repeat=N))
    hidden_patterns = list(product([-1, 1], repeat=M))
    Z = 0
    for v in visible_patterns:
        for h in hidden_patterns:
            Z += math.exp(-H(W, h, v, Theta_v, Theta_h))
    return math.exp(-H(W, h_vec, v_vec, Theta_v, Theta_h)) / Z

# Compute marginal distribution P(v) for XOR patterns
visible_patterns = list(product([-1, 1], repeat=N))
hidden_patterns = list(product([-1, 1], repeat= M ))

print("\nMarginal Boltzmann probabilities for XOR patterns:")
model_dist = {}

for pattern in xor_data:
    prob_v = 0
    for h in hidden_patterns:
        prob_v += boltzmann_dist(W, h, pattern, Theta_v, Theta_h)
    
    # Convert to clean Python tuple of ints
    clean_pattern = tuple(int(x) for x in pattern)
    model_dist[clean_pattern] = prob_v
    print(f"P(v={clean_pattern}) = {prob_v:.6f}")