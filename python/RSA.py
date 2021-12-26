import time
import numpy as np
from solution import Solution


def RSA(f_obj, lb, ub, dim, n, max_iter):
    Convergence_curve = np.zeros(max_iter)
    s = Solution()
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")


    best_p = np.zeros((1, dim))  # best positions
    best_f = np.inf  # best fitness

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    x = np.zeros((n, dim))  # Initialize the positions of solution

    for i in range(dim):
        x[:, i] = (
                np.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]
        )
    x_new = np.zeros((n, dim))

    t = 0  # starting iteration
    alpha = 0.1  # the best value 0.1
    beta = 0.005  # the best value 0.005
    f_fun = np.zeros((1, n))  # old fitness values
    f_fun_new = np.zeros((1, n))  # new fitness values

    for i in range(n):
        f_fun[0, i] = f_obj(x[i])  # Calculate the fitness values of solutions
        s.funcexec += 1
        if f_fun[0, i] < best_f:
            best_f = f_fun[0, i]
            best_p = x[i]

    while t < max_iter:  # Main loop Update the Position of solutions
        es = 2 * np.random.randn() * (1 - (t / max_iter))
        for i in range(1, n):
            for j in range(dim):
                r = best_p[j] - x[np.random.choice([0, n - 1]), j] / (best_p[j] + np.spacing(1))
                p = alpha + (x[i, j] - np.mean(x[i])) / (best_p[j] * (ub[j] - lb[j]) + np.spacing(1))
                eta = best_p[j] * p
                if t < 0.25 * max_iter:
                    x_new[i, j] = best_p[j] - eta * beta - r * np.random.rand()
                elif 0.5 * max_iter > t >= 0.25 * max_iter:
                    x_new[i, j] = best_p[j] * x[np.random.choice([0, n - 1]), j] * es * np.random.rand()
                elif 0.75 * max_iter > t >= 0.5 * max_iter:
                    x_new[i, j] = best_p[j] * p * np.random.rand()
                else:
                    x_new[i, j] = best_p[j] - eta * np.spacing(1) - r * np.random.rand()

            x_new[i] = np.clip(x_new[i], a_min=lb, a_max=ub)
            f_fun_new[0, i] = f_obj(x_new[i])
            s.funcexec += 1
            if f_fun_new[0, i] < f_fun[0, i]:
                x[i] = x_new[i]
                f_fun[0, i] = f_fun_new[0, i]
            if f_fun[0, i] < best_f:
                best_f = f_fun[0, i]
                best_p = x[i]

        print(f"At iteration {t} the best solution fitness is: {best_f}, {best_p}")
        Convergence_curve[t] = best_f
        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "RSA"
    s.objfname = f_obj.__name__
    s.best = best_p
    s.no_particles = n
    s.max_iter = max_iter

    return s
