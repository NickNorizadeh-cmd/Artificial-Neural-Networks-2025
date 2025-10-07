import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('training_set.csv')
training_set = np.asarray(training_set)
validation_set = pd.read_csv('validation_set.csv')
validation_set = np.asarray(validation_set)

def train_perceptron(patterns, eta, nu_max, M1, M2):

    w_1, theta_1 = np.random.normal(0, np.sqrt(1/M1), size = [M1,2]), np.zeros(M1)
    w_2, theta_2 = np.random.normal(0, np.sqrt(1/M2), size = [M2,M1]), np.zeros(M2)
    w_3,theta_3 = np.random.normal(0, 1, size = M2), np.zeros(1)
    hidden_layer_1, hidden_layer_2 = np.zeros(M1), np.zeros(M2)

    for nu in range(nu_max):

        indices = np.arange(patterns.shape[0])
        np.random.shuffle(indices)
        for mu in indices:
            input = patterns[mu,:2]
            b_1 = w_1 @ input - theta_1
            hidden_layer_1 = np.tanh(b_1)
            b_2 = w_2 @ hidden_layer_1 - theta_2
            hidden_layer_2 = np.tanh(b_2)
            b_3 = w_3 @ hidden_layer_2 - theta_3
            output = np.tanh(b_3)
    
            deltas = [0 for i in range(3)]
            deltas[-1] = (1 - np.tanh(b_3)**2) * (patterns[mu,2] - output) # delta_3
            deltas[-2] = (1 - np.tanh(b_2)**2) * (deltas[-1] * w_3) # delta_2
            deltas[-3] = (1 - np.tanh(b_1)**2) * (np.transpose(w_2) @ deltas[-2]) # delta_1
    
            theta_3 += -eta * deltas[-1]
            w_3 += eta * deltas[-1] * hidden_layer_2
            theta_2 += -eta * deltas[-2]
            nrows, ncols = w_2.shape
            for m in range(nrows):
                for n in range(ncols):
                    w_2[m,n] += eta * deltas[-2][m] * hidden_layer_1[n]
            theta_1 += -eta * deltas[-3]
            nrows, ncols = w_1.shape
            for m in range(nrows):
                for n in range(ncols):
                    w_1[m,n] += eta * deltas[-3][m] * input[n]

    return w_1, w_2, w_3, theta_1, theta_2, theta_3


def output(input, w_1, w_2, w_3, theta_1, theta_2, theta_3):

    b_1 = w_1 @ input - theta_1
    V_1 = np.tanh(b_1)
    b_2 = w_2 @ V_1 - theta_2
    V_2 = np.tanh(b_2)
    b_3 = b_2 = w_3 @ V_2 - theta_3
    output = np.tanh(b_3)

    return output

eta = 0.01
nu_max = 200
M1 = 6
M2 = 6

w_1, w_2, w_3, theta_1, theta_2, theta_3 = train_perceptron(training_set, eta, nu_max, M1, M2)

pval = validation_set.shape[0]
outputs = np.array([output(validation_set[i,:2] ,w_1, w_2, w_3, theta_1, theta_2, theta_3) for i in range(pval)])
output_signs = np.sign(outputs)
output_errors = np.abs(output_signs - np.reshape(validation_set[:,2], output_signs.shape))
validation_error = np.sum(output_errors)
validation_error/= (2*pval)
print(validation_error)