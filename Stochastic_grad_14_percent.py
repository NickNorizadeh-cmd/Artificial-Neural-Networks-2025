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
X_std = (X_raw - mean) / std

# Rebalance training data
X_neg = X_std[T_raw == -1]
X_pos = X_std[T_raw == 1]
np.random.seed(42)
idx = np.random.choice(len(X_neg), len(X_pos), replace=False)
X_bal = np.vstack((X_neg[idx], X_pos))
T_bal = np.hstack((-1 * np.ones(len(X_pos)), 1 * np.ones(len(X_pos))))
perm = np.random.permutation(len(X_bal))
X = X_bal[perm]
T = T_bal[perm]

# Load and standardize validation data
val_data = pd.read_csv("validation_set.csv").values
X_val = (val_data[:, :2] - mean) / std
T_val = val_data[:, 2]

print("Training class balance (balanced):", np.unique(T, return_counts=True))
print("Validation class balance:", np.unique(T_val, return_counts=True))

# Hyperparameters
M1, M2 = 6, 6
lr = 0.01
epochs = 500

# Xavier initialization for tanh
w1 = np.random.randn(M1, 2) * np.sqrt(1 / 2)
w2 = np.random.randn(M2, M1) * np.sqrt(1 / M1)
w3 = np.random.randn(M2, 1) * np.sqrt(1 / M2)
t1 = np.zeros((M1, 1))
t2 = np.zeros((M2, 1))
t3 = np.zeros(1)


# Activation functions
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2

# Prediction function
def predict(X_input):
    V1 = tanh(-t1 + w1 @ X_input.T)
    V2 = tanh(-t2 + w2 @ V1)
    O = tanh(-t3 + (w3.T @ V2)).reshape(-1)
    return np.where(O >= 0, 1, -1)

# Early stopping setup
best_C = float('inf')
patience = 20
wait = 0

# Training loop
N = X.shape[0]
for epoch in range(epochs):
    for i in range(N):
        x_i = X[i].reshape(-1, 1)
        t_i = T[i]

        V1 = tanh(-t1 + w1 @ x_i)
        V2 = tanh(-t2 + w2 @ V1)
        O = tanh(-t3 + (w3.T @ V2)).item()

        error = O - t_i
        
        delta3 = error * dtanh(-t3 + (w3.T @ V2)).item()
        
        dw3 = lr * delta3 * V2
        dt3 = -lr* delta3

        delta2 = (delta3 * w3) * dtanh(-t2 + w2 @ V1)
        dw2 = lr * delta2 @ V1.T # dw2 has dimension M2x1 * 1xM1 = M2xM1
        dt2 = -lr* delta2

        delta1 = (w2.T @ delta2) * dtanh(-t1 + w1 @ x_i)
        dw1 = lr * delta1 @ x_i.T
        dt1 = -lr* delta1

        w3 -= dw3
        t3 -= dt3
        w2 -= dw2
        t2 -= dt2
        w1 -= dw1
        t1 -= dt1

    # Learning rate decay
    if epoch % 50 == 0 and epoch > 0:
        lr *= 0.5
        print(f"🔧 Learning rate decayed to {lr:.5f}")

    # Monitoring
    if epoch % 5 == 0:
        train_preds = predict(X)
        train_acc = np.mean(train_preds == T)
        val_preds = predict(X_val)
        C = (1 / (2 * len(X_val))) * np.sum(np.abs(val_preds - T_val))
        print(f"Epoch {epoch}: Training accuracy = {train_acc:.4f}, Validation error C = {C:.4f}")

        if C < best_C:
            best_C = C
            wait = 0
        else:
            wait += 5

        if best_C < 0.12:
            print(f"✅ Early stopping: validation error below 12% (C = {best_C:.4f})")
            break

        #if wait >= patience:
           # print("⏹️ Early stopping: no improvement in validation error")
            #break

# Diagnostic printout
print("\n🔍 Sample Predictions vs Targets:")
for i in range(min(10, len(X_val))):
    x = X_val[i].reshape(1, -1)
    pred = predict(x)
    print(f"Target: {T_val[i]}, Predicted: {pred[0]}")

# Decision boundary plot
def plot_decision_boundary(X, T):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(grid)
    Z = preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap="coolwarm", edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X, T)

# Save weights
np.savetxt("w1.csv", w1, delimiter=",")
np.savetxt("w2.csv", w2, delimiter=",")
np.savetxt("w3.csv", w3.T, delimiter=",")
np.savetxt("t1.csv", t1.T, delimiter=",")
np.savetxt("t2.csv", t2.T, delimiter=",")
np.savetxt("t3.csv", np.array([t3]), delimiter=",")
