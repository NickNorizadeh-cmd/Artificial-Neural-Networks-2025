import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


training_set = pd.read_csv('training_set.csv')
training_set = np.asarray(training_set)
validation_set = pd.read_csv('validation_set.csv')
validation_set = np.asarray(validation_set)

#print(training_set)

M1 = 6
M2 = 6
nu_max = 130
eta = 0.01

# Activation functions
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2


def training_algorithm (patterns, eta, nu_max, M1, M2):
    w1 = np.random.randn(M1, 2) 
    w2 = np.random.randn(M2, M1)
    w3 = np.random.randn(M2)
    t1 = np.zeros(M1)  
    t2 = np.zeros(M2)
    t3 = np.zeros(1)
    V1 = np.zeros(M1)
    V2 = np.zeros(M2)
    
    for nu in range(nu_max):
        #print(" Current Nu value", nu)
        mu_list = np.arange(patterns.shape[0]) # Get a list of the rows in the training set and call them mu_list
        np.random.shuffle(mu_list)
        for mu in mu_list:
                #print(" Current Mu value", mu)
                input = patterns[mu,:2]
                b_1 = w1 @ input - t1
                #print("b1 is: ", b_1.shape)
                V1 = np.tanh(b_1)
                b_2 = w2 @ V1 - t2
                #print("Shape of V1", V1.shape)
                #print("Shape of w2", w2.shape)
                V2 = np.tanh(b_2)
                #print("Shape of V2", V2.shape)
                b_3 = V2 @ w3 - t3
                output = np.tanh(b_3)

                output_error = patterns[mu,2] -  output
                
                delta_3 = output_error * dtanh(b_3)
                #print("Shape of delta_3", delta_3.shape)
                delta_2 = delta_3 * w3 * dtanh(b_2) # delta 3 is a scalar
                #print("Shape of delta_2", delta_2.shape)
                #print("Shape of W2 transpose", w2.T.shape)
                delta_1 = w2.T @ delta_2 * dtanh(b_1)
                #print("Shape of delta1", delta_1.shape)
                
                w3 += eta * delta_3 * V2
                w2 += eta * delta_2 * V1
                number_of_rows = w1.shape[0]
                for m in range(number_of_rows):
                     w1[m,:] += eta * delta_1[m] * input
                #w1 += eta * delta_1 * input

                t3 -= eta* delta_3
                t2 -= eta* delta_2
                t1 -= eta* delta_1


    return w1,w2,w3,t1,t2,t3


def output(input, w_1, w_2, w_3, theta_1, theta_2, theta_3):

    b_1 = w_1 @ input - theta_1
    V_1 = np.tanh(b_1)
    b_2 = w_2 @ V_1 - theta_2
    V_2 = np.tanh(b_2)
    b_3 = b_2 = w_3 @ V_2 - theta_3
    output = np.tanh(b_3)

    return output


w_1, w_2, w_3, theta_1, theta_2, theta_3 = training_algorithm(training_set, eta, nu_max, M1, M2)

pval = validation_set.shape[0]
outputs = np.array([output(validation_set[i,:2] ,w_1, w_2, w_3, theta_1, theta_2, theta_3) for i in range(pval)])
output_signs = np.sign(outputs)
output_errors = np.abs(output_signs - np.reshape(validation_set[:,2], output_signs.shape))
validation_error = np.sum(output_errors)
validation_error/= (2*pval)
print("The validation error is", validation_error)




        

