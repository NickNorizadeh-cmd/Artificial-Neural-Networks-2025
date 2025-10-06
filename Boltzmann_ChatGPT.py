import numpy as np
import matplotlib.pyplot as plt

# XOR dataset: 4 patterns with equal probability
xor_data = np.array([
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
    [-1, -1, -1]
])

# Activation function for ±1 neurons
def prob_plus1(x):
    return 0.5 * (1 + np.tanh(x))

def sample_pm1(prob):
    return np.where(np.random.rand(*prob.shape) < prob, +1, -1)

# RBM class for ±1 neurons
class RBMpm1:
    def __init__(self, n_visible, n_hidden, learning_rate=0.05):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.W = np.random.normal(0, 0.1, (n_hidden, n_visible))
        self.theta_h = np.zeros(n_hidden)
        self.theta_v = np.zeros(n_visible)

    def cd_k(self, data, k=1, epochs=5000):
        for epoch in range(epochs):
            v0 = data[np.random.randint(0, len(data))]
            b_h0 = self.W @ v0 - self.theta_h
            p_h0 = prob_plus1(b_h0)
            h0 = sample_pm1(p_h0)

            vk = v0.copy()
            hk = h0.copy()

            for step in range(k):
                b_vk = self.W.T @ hk - self.theta_v
                p_vk = prob_plus1(b_vk)
                vk = sample_pm1(p_vk)

                b_hk = self.W @ vk - self.theta_h
                p_hk = prob_plus1(b_hk)
                hk = sample_pm1(p_hk)

            self.W += self.lr * (np.outer(np.tanh(b_h0), v0) - np.outer(np.tanh(b_hk), vk))
            self.theta_h -= self.lr * (np.tanh(b_h0) - np.tanh(b_hk))
            self.theta_v -= self.lr * (np.tanh(self.W.T @ h0 - self.theta_v) - np.tanh(self.W.T @ hk - self.theta_v))

    def estimate_distribution(self, n_samples=10000):
        counts = {}
        for _ in range(n_samples):
            v = xor_data[np.random.randint(0, len(xor_data))]
            b_h = self.W @ v - self.theta_h
            h = sample_pm1(prob_plus1(b_h))

            b_v = self.W.T @ h - self.theta_v
            v_sample = sample_pm1(prob_plus1(b_v))

            key = tuple(v_sample)
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

# KL divergence between true and model distributions
def kl_divergence(true_dist, model_dist):
    kl = 0.0
    for pattern in true_dist:
        p = true_dist[pattern]
        q = model_dist.get(pattern, 1e-12)
        kl += p * np.log(p / q)
    return kl

# Theoretical bound from Equation (4.40)
def theoretical_bound(M, N=3):
    threshold = 2**(N - 1) - 1
    if M >= threshold:
        return 0.0
    log_term = np.floor(np.log2(M + 1))
    bound = np.log(2) * (N - log_term - (M + 1) / (2**log_term))
    return bound

# True XOR distribution
true_dist = {tuple(p): 0.25 for p in xor_data}

# Run multiple trials per M
hidden_units = [1, 2, 4, 8]
n_trials = 10  # Number of runs per M
kl_results = {M: [] for M in hidden_units}
theory_values = []

for M in hidden_units:
    theory_values.append(theoretical_bound(M))
    for _ in range(n_trials):
        rbm = RBMpm1(n_visible=3, n_hidden=M, learning_rate=0.05)
        rbm.cd_k(xor_data, k=1, epochs=5000)
        model_dist = rbm.estimate_distribution()
        kl = kl_divergence(true_dist, model_dist)
        kl_results[M].append(kl)


plt.figure(figsize=(8, 5))

# Scatter plot for empirical KL values
for i, M in enumerate(hidden_units):
    plt.scatter([M]*n_trials, kl_results[M], alpha=0.6, label=f'M={M}' if i == 0 else "")

# Line plot for theoretical bound
plt.plot(hidden_units, theory_values, linestyle='--', color='black', label='Theory (Eq. 4.40)')

plt.xlabel('Number of Hidden Neurons (M)')
plt.ylabel('KL Divergence')
plt.title('KL Divergence vs. Hidden Units (Multiple Runs)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
