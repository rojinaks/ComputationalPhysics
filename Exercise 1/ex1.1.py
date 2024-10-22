# %% Generate Data
import numpy as np
import matplotlib.pyplot as plt
import main


P = 10_000  # Number of points

x, y = np.random.uniform(-1, 1, size=(2, P))  # Generate x and y simultaneously
radius_squared = x ** 2 + y ** 2
circle = radius_squared <= 1  # Points inside the circle

pi = 4 * np.mean(circle)
pi_std = 4 * np.std(circle)

# Output the estimate of π
print(f'π = {pi:.3f} ± {pi_std:.3f}')

# %% Plot Radii in Histogram
bins = [i * 0.1 for i in range(15)]
r = np.sqrt(radius_squared)
plt.hist(r, bins=bins)
plt.xlabel('Radius')
plt.ylabel("Radius Count")
plt.savefig(main.FIGS_DIR/"ex1.1_radii_hist.pdf")
plt.show()

# %% Plot for the Radii Squared in Histogram
plt.hist(radius_squared, bins=bins)
plt.xlabel('Radius Squared')
plt.ylabel('Radius Squared Count')
plt.savefig(main.FIGS_DIR/"ex1.1_radii_squared_hist.pdf")
plt.show()

# %% Histogram of the indicator variable 4[x² + y² ≤ 1]
plt.hist(4 * circle, bins=25, alpha=0.75, edgecolor='black', zorder=100)
plt.axvline(pi, color='red', linestyle='--', label=r'calculated $\pi$ Value')
plt.axvline(np.pi, color='black', linestyle=':', label=r'true $\pi$ Value')
plt.axvspan(pi - pi_std, pi + pi_std, alpha=0.25, color='blue', zorder=0,
            label=r'$1\sigma$-interval')
plt.legend()
plt.xlabel(r'$4[x^2 + y^2 \leq 1]$')
plt.ylabel("Value Count")
plt.savefig(main.FIGS_DIR/"ex1.1_indicator_hist.pdf")
plt.show()

