from RSA import RSA
import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sum(x ** 2)

lower_bound = -10
upper_bound = 10
dimension = 10
no_particles = 30
max_iter = 100

opt = RSA(objective_function, lower_bound, upper_bound, dimension, no_particles, max_iter)
print(
    f"{opt.optimizer} algorithm\nbest: {opt.best}\nexecution time: {opt.executionTime}\nvalue: {opt.convergence[-1]}\nfunexec: {opt.funcexec}")
plt.plot(opt.convergence, color="r", label="RSA")

plt.legend(["RSA"], loc="upper right")
plt.title("optimization using RSA")
plt.xlabel("Number of analysis")
plt.ylabel("Weight(Kg)")
plt.show()