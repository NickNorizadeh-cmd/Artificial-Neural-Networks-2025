import numpy as np
import pandas as pd
training_data = pd.read_csv('training_set.csv')
validation_data = pd.read_csv('validation_set.csv')
# to display the first 5 lines of loaded data
#print(training_data.head())

def generate_weight_matrices(m1, m2):
    # initialize with Gaussian(0,1) — new values each time
    W_1 = np.random.randn(m1, 2)      # (m1, 2)
    W_2 = np.random.randn(m2, m1)     # (m2, m1)
    W_3 = np.random.randn(m2, 1)      # (m2, 1)

    # Print shapes
    #print("W_1 shape:", W_1.shape)
    #print("W_2 shape:", W_2.shape)
    #print("W_3 shape:", W_3.shape)

    return W_1, W_2, W_3

# Example usage
W1, W2, W3 = generate_weight_matrices(4, 3)


def generate_threshold_matrices(m1, m2):
    # initialize thresholds (biases) with zeros
    Theta_1 = np.zeros((m1, 1))   # (m1, 1) column vector
    Theta_2 = np.zeros((m2, 1))   # (m2, 1) column vector
    Theta_3 = 0    # scalar as a 1x1 array (or just 0)

    # Print shapes
    print("Theta_1 shape:", Theta_1.shape)
    print("Theta_2 shape:", Theta_2.shape)
    print("Theta_3 shape:", Theta_3)

    return Theta_1, Theta_2, Theta_3


#Main program
W1, W2, W3 = generate_weight_matrices(2, 2)
Theta1, Theta2, Theta3 = generate_threshold_matrices(2, 2)
for nu in training_data:
    