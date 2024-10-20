import numpy as np
import matplotlib.pyplot as plt

P = 10000  # Number of points
X = 1  # Number of simulations

pi_values = []

for _ in range(X):
    x, y = np.random.uniform(-1, 1, size=(2, P))  # Generate x and y simultaneously
    rad2 = x ** 2 + y ** 2
    circle = rad2 <= 1  # Points inside the circle

    pi_mean = 4 * np.mean(circle)
    pi_values.append(pi_mean)

# Convert to NumPy array
pi_values = np.array(pi_values)
pi_std = np.std(pi_values)

# Output the estimate of π
print(f'π = {np.mean(pi_values)} ± {pi_std}')

# Histogram of π estimates
plt.hist(pi_values, bins=25, alpha=0.75, edgecolor='black')
plt.axvline(np.pi, color='black', linestyle=':', label='')
plt.legend()
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()

# Plot for the radius
r = np.sqrt(rad2)
plt.hist(r, bins=25, edgecolor='black')
plt.xlabel('Radius')
plt.ylabel(' ')
plt.title('')
plt.show()

# Plot for the radius squared
plt.hist(rad2, bins=25, edgecolor='black')
plt.xlabel('Radius Squared')
plt.ylabel('')
plt.title('')
plt.show()

# Histogram of the indicator variable 4[x² + y² ≤ 1]
plt.hist(4 * circle, bins=25, alpha=0.75, edgecolor='black')
plt.axvline(np.mean(4 * circle), color='red', linestyle='--', label='mean')
plt.axvline(np.pi, color='black', linestyle=':', label='true π Value')
plt.axvspan(np.mean(4 * circle) - pi_std, np.mean(4 * circle) + pi_std, alpha=0.25, color='blue',
            label='mean ± std Dev')
plt.legend()
plt.xlabel('4[x² + y² ≤ 1]')
plt.ylabel('')
plt.title('')
plt.show()
