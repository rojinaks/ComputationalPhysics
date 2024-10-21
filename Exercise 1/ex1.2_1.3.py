import matplotlib.pyplot as plt
import numpy as np

import main


for pairs, experiments in [(100, 100), (1, 10_000)]:
    points = main.generate_points(pairs, experiments)
    pi = main.calculate_pi(points=points)

    pi_mean = pi.mean()
    pi_std = pi.std()

    print(f"Calculated π = {pi_mean:.3f} +- {pi_std:.3f}")
    print(f"Real π: {np.pi:.3f}\n")

    main.generate_hist_for_pi(pi, pairs, experiments)

