import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
data = pd.read_csv("training_set.csv").values
X_raw = data[:, :2]
T_raw = data[:, 2]

# Standardize training inputs
mean = X_raw.mean(axis=0)
std = X_raw.std(axis=0)
X = (X_raw - mean) / std
T = T_raw

# Load and standardize validation data
val_data = pd.read_csv("validation_set.csv").values
X_val = (val_data[:, :2] - mean) / std
T_val = val_data[:, 2]

print("Training class balance:", np.unique(T, return_counts=True))
print("Validation class balance:", np.unique(T_val, return_counts=True))

# Activation functions
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2

# Prediction function
def predict(X_input, w1, w2, w3, t1, t2, t3):
    V1 = tanh(-t1 + w1 @ X_input.T)
    V2 = tanh(-t2 + w2 @ V1)
    O = tanh(-t3 + (w3.T @ V2)).reshape(-1)
    return np.where(O >= 0, 1, -1)

# Training function
def train_and_evaluate(M1, M2, lr=0.01, epochs=500):
    # Xavier initialization
    w1 = np.random.randn(M1, 2) * np.sqrt(1 / 2)
    w2 = np.random.randn(M2, M1) * np.sqrt(1 / M1)
    w3 = np.random.randn(M2, 1) * np.sqrt(1 / M2)
    t1 = np.zeros((M1, 1))
    t2 = np.zeros((M2, 1))
    t3 = np.zeros(1)

    for epoch in range(epochs):
        for i in range(X.shape[0]):
            x_i = X[i].reshape(-1, 1)
            t_i = T[i]

            V1 = tanh(-t1 + w1 @ x_i)
            V2 = tanh(-t2 + w2 @ V1)
            O = tanh(-t3 + (w3.T @ V2)).item()

            error = O - t_i
            delta3 = error * dtanh(-t3 + (w3.T @ V2)).item()

            dw3 = lr * delta3 * V2
            dt3 = -lr * delta3

            delta2 = (delta3 * w3) * dtanh(-t2 + w2 @ V1)
            dw2 = lr * delta2 @ V1.T
            dt2 = -lr * delta2

            delta1 = (w2.T @ delta2) * dtanh(-t1 + w1 @ x_i)
            dw1 = lr * delta1 @ x_i.T
            dt1 = -lr * delta1

            w3 += dw3
            t3 += dt3
            w2 += dw2
            t2 += dt2
            w1 += dw1
            t1 += dt1

    val_preds = predict(X_val, w1, w2, w3, t1, t2, t3)
    C = (1 / (2 * len(X_val))) * np.sum(np.abs(val_preds - T_val))
    return C, (w1, w2, w3, t1, t2, t3)

# Sweep over configurations
configs = [(6,6), (10,8), (12,8), (20,10), (12,4), (6,12)]
results = {}

for M1, M2 in configs:
    print(f"\n🔍 Testing M1={M1}, M2={M2}")
    C, weights = train_and_evaluate(M1, M2)
    results[(M1, M2)] = (C, weights)
    print(f"Validation error C = {C:.4f}")

# Best result
best_config = min(results, key=lambda k: results[k][0])
best_C, (w1, w2, w3, t1, t2, t3) = results[best_config]
print(f"\n🏆 Best configuration: M1={best_config[0]}, M2={best_config[1]} with C = {best_C:.4f}")

# Diagnostic printout
print("\n🔍 Sample Predictions vs Targets:")
final_preds = predict(X_val, w1, w2, w3, t1, t2, t3)
for i in range(min(10, len(X_val))):
    print(f"Target: {T_val[i]}, Predicted: {final_preds[i]}")

# Decision boundary plot
def plot_decision_boundary(X, T, w1, w2, w3, t1, t2, t3):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(grid, w1, w2, w3, t1, t2, t3)
    Z = preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap="coolwarm", edgecolors='k')
    plt.title(f"Decision Boundary (M1={best_config[0]}, M2={best_config[1]})")
    plt.show()

plot_decision_boundary(X, T, w1, w2, w3, t1, t2, t3)

# Save weights
np.savetxt("w1.csv", w1, delimiter=",")
np.savetxt("w2.csv", w2, delimiter=",")
np.savetxt("w3.csv", w3.T, delimiter=",")
np.savetxt("t1.csv", t1.T, delimiter=",")
np.savetxt("t2.csv", t2.T, delimiter=",")
np.savetxt("t3.csv", np.array([t3]), delimiter=",")
