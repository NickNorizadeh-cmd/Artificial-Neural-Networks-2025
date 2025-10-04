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
M = hidden_units[1]
v_max = 10000  # Reduced for faster testing
p0 = 20
k = 10
eta = 0.005

def p(b_i):
    return 1 / (1 + np.exp(-2 * b_i))

# KL divergence and theoretical bound functions
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

# Store KL and theory values
kl_values = []
theory_values = []

#for M in hidden_units:
print(f"\nTraining RBM with {M} hidden units...")
N = 3  # number of visible neurons
W = np.random.randn(M, N)
Theta_h = np.zeros((M, 1))
Theta_v = np.zeros((N, 1))
error_list = []

# CD-k algorithm
for epoch in range(v_max):
    x_sample = np.random.choice(range(xor_data.shape[0]), size=p0)
    delta_w_m_n = np.zeros(W.shape)
    delta_theta_v = np.zeros((N, 1))
    delta_theta_h = np.zeros((M, 1))

    for mu in range(p0):
        v_vec = xor_data[x_sample[mu]].reshape((N, 1))
        b_h = W @ v_vec - Theta_h
        h_vec = np.zeros((M, 1))

        for i in range(M):
            r = random.random()
            prob = p(b_h[i])
            h_vec[i] = 1 if r < prob else -1

        b_h_0 = b_h.copy() # save the arguments that are b^h(0)
        v_vec_0 = v_vec.copy()  # save the arguments that are v_n(0)

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

def energy(v_vec, h_vec, W, Theta_v, Theta_h):
    v_vec = np.array(v_vec).reshape((N, 1))
    h_vec = np.array(h_vec).reshape((M, 1))
    return - (h_vec.T @ W @ v_vec + Theta_v.T @ v_vec + Theta_h.T @ h_vec).item()

def compute_model_distribution(W, Theta_v, Theta_h, N, M):
    visible_patterns = list(product([-1, 1], repeat=N))
    hidden_patterns = list(product([-1, 1], repeat=M))

    Z = 0
    P_v = {}

    for v in visible_patterns:
        v_energy_sum = 0
        for h in hidden_patterns:
            v_energy_sum += math.exp(-energy(v, h, W, Theta_v, Theta_h))
        P_v[v] = v_energy_sum
        Z += v_energy_sum

    # Normalize
    for v in P_v:
        P_v[v] /= Z

    return P_v

print (compute_model_distribution(W,Theta_v,Theta_h,N,M))


# Sampling to estimate model distribution
#iterates = 100

# Initialize v_vec and h_vec for sampling
#v_vec = xor_data[np.random.randint(0, len(xor_data))].reshape((N, 1))
#b_h = W @ v_vec - Theta_h
#h_vec = np.where(np.random.rand(M, 1) < p(b_h), 1, -1)

#count1 = 0
#count2 = 0
#count3 = 0
#count4 = 0
#count_not_in_xor = 0

#for t in range(iterates):
#    if (np.array_equal(v_vec.flatten(),xor_data[0])):
#        count1 += 1
#    elif(np.array_equal(v_vec.flatten(),xor_data[1])):
#        count2 += 1
#    elif(np.array_equal(v_vec.flatten(),xor_data[2])):
#        count3 += 1
#    elif(np.array_equal(v_vec.flatten(),xor_data[3])):
#        count4 += 1
#    else:
#        count_not_in_xor += 1 # With this it will count all the cases that don't belong to XOR


#    b_v = W.T @ h_vec - Theta_v
#    for j in range(N):
#        r = random.random()
#        prob = p(b_v[j])
#       v_vec[j] = 1 if r < prob else -1
    
#    print("Sampled v_vec:", v_vec.flatten())

#    b_h = W @ v_vec - Theta_h
#    for i in range(M):
#        r = random.random()
#        prob = p(b_h[i])
#        h_vec[i] = 1 if r < prob else -1

#fraction1 = count1/iterates
#fraction2 = count2/iterates
#fraction3 = count3/iterates
#fraction4 = count4/iterates
#fraction5 = count_not_in_xor/iterates


#print(f"\nResults for M = {M}:")
#print(f"Pattern 1: {fraction1:.2f}")
#print(f"Pattern 2: {fraction2:.2f}")
#print(f"Pattern 3: {fraction3:.2f}")
#print(f"Pattern 4: {fraction4:.2f}")
#print(f"Non-XOR patterns: {fraction5:.2f}")




