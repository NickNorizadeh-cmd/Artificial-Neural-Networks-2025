import numpy as np
import random
import matplotlib.pyplot as plt

# XOR dataset: 4 patterns with equal probability
xor_data = np.array([
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
    [-1, -1, -1]
])
hidden_units = [1, 2, 4, 8]
v_max = 10**4
p0 = 20
k = 10
eta = 0.005

M = hidden_units[0] # number of hidden neurons
N = 3 # nr of visible neurons
W = np.random.randn(M,N)
#print("W is ",W)
Theta_h = np.zeros((M, 1)) # depend on nr of hidden neurons
Theta_v = np.zeros((3, 1))


def p(b_i):
    return 1 / (1 + np.exp(-2*b_i))


for i in range (v_max):

    x_sample = np.random.choice(range(xor_data.shape[0]), size=p0) # x is our sampled input data consisiting of 20 random numbers in the range 0 to 3

    delta_w_m_n = np.zeros(W.shape)
    delta_theta_n = np.zeros(Theta_h.shape)
    delta_theta_m = np.zeros(Theta_v.shape)
    delta_theta_v = np.zeros(Theta_h.shape)
    delta_theta_h = np.zeros(Theta_v.shape)

    for mu in range (p0):
        v_vec = xor_data[x_sample[mu]]
        print(v_vec)

        b_h = W@v_vec - Theta_h
        h_vec = np.zeros((M,1))
        for i in range(M):
            r = random.randint(0,1)
            prob = p(b_h[i]) 
           # print("h_vec is", h_vec)
            if r < prob:
                h_vec[i] = 1
            else: # if r is greater than p, meaning it's in the 1-p region
                h_vec[i] = -1

        b_h_0 = b_h.copy()
        v_vec_0 = v_vec.copy() # at time step 0, there is only one value
        print(v_vec_0)

        for t in range (k):
            b_v  = W.T@ h_vec - Theta_v
           #print(b_h.shape)
           #print("Shape of product", b_v.shape)
            
            for j in range(N):
                r = random.randint(0,1)
                prob = p(b_v[j])
                if r < prob:
                    v_vec[j] = 1
                else:
                    v_vec[j] = -1
            
            b_h = W @ v_vec- Theta_h
            for i in range(M):
                r = random.randint(0,1)
                prob = p(b_h[i])
               # print(b_h.shape,)
                if r < prob:
                    h_vec[i] = 1
                else: # if r is greater than p, meaning it's in the 1-p region
                    h_vec[i] = -1
            
        first_term =  np.outer(np.tanh(b_h_0),v_vec_0)
        second_term = np.outer(np.tanh(b_h),v_vec)
        delta_w_m_n += eta * (first_term - second_term)

        delta_theta_v -= -eta*(v_vec_0 - v_vec)
        print("eta", eta.shape)
        print("v_vec shape", v_vec.shape)
        print("v_vec0_shape", v_vec_0.shape)                    

        delta_theta_h -= -eta*(np.tanh(b_h_0) - np.tanh(b_h))

    W += delta_w_m_n
    Theta_h += delta_theta_h
    Theta_v += delta_theta_v

