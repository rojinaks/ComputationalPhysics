import numpy as np
import matplotlib.pyplot as plt

dimension = 2
n_experiments = 1
n_points_per_experiment = 10000

points = np.random.uniform(low=-1.0, high=1.0,
                           size=(dimension, n_points_per_experiment, n_experiments,))
# x = np.random.uniform(low=-1.0, high=1.0,
#                           size=(dimension, n_points_per_experiment, n_experiments, ))

radius = (points ** 2).sum(axis=0)
inside_the_circle = (radius <= 1)
inside_the_square = ((-1.0 <= points[0]) & (points[0] <= 1.0) &
                     (-1.0 <= points[1]) & (points[1] <= 1.0))

pi_by_4 = inside_the_circle.sum(axis=0) / inside_the_square.sum(axis=0)
pi = 4 * pi_by_4
print(pi)

average = pi.mean()
uncertainty = pi.std()
# print(f'π={average} ± {uncertainty}')

r_2 = radius ** 2
kreis = r_2 <= 1
print(f'π={average} ± {uncertainty}')  # aus vorlesung

# plt.hist(pi, bins=25)
# plt.axvline(np.pi, color='black', zorder=1, linestyle=':')
# plt.axvspan(average-uncertainty, average+uncertainty, alpha=0.25)
# plt.show()


# ------------------------------------------------------------------------------------


P = 10000
X = 1

pi_wert = []

for _ in range(X):
    x2 = np.random.uniform(-1, 1, size=P)
    y2 = np.random.uniform(-1, 1, size=P)

    rad2 = x2 ** 2 + y2 ** 2
    kreis2 = rad2 <= 1

    pi_mean2 = 4 * np.mean(kreis2)
    pi_wert.append(pi_mean2)

pi_wert = np.array(pi_wert)
pi_std2 = np.std(pi_wert)

print(f'π={np.mean(pi_wert)} ± {pi_std2}')


plt.hist(pi_wert, bins=25)
plt.axvline(np.pi, color='black', zorder=1, linestyle=':')
plt.axvspan(np.mean(pi_wert) - pi_std2, np.mean(pi_wert) + pi_std2, alpha=0.25)
plt.show()

r = np.sqrt(rad2)
plt.figure()
plt.hist(r, bins=25, edgecolor='black')
plt.show()

r2 = rad2
plt.figure()
plt.hist(r2, bins=25, edgecolor='black')
plt.show()

# -------------------------------------------------------------------------------
P = 10000
X = 1

pi_wert = []

for _ in range(X):
    x21 = np.random.uniform(-1, 1, size=P)
    y21 = np.random.uniform(-1, 1, size=P)


    rad2 = x21 ** 2 + y21 ** 2


    kreis3 = (rad2 <= 1)


    pi_mean3 = 4 * np.mean(kreis3)

    pi_wert.append(pi_mean3)

pi_wert = np.array(pi_wert)
pi_std3 = np.std(pi_wert)


print(f'π={np.mean(pi_wert)} ± {pi_std3}')


plt.hist(pi_wert, bins=25)
plt.axvline(np.pi, color='black', zorder=1, linestyle=':')
plt.axvspan(np.mean(pi_wert) - pi_std3, np.mean(pi_wert) + pi_std3, alpha=0.25)
plt.show()






