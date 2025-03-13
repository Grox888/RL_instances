import random

import numpy as np


def f(x):
    return x ** 3 - 5


def solve_function(f):
    error = 1e-5
    x = 2
    n = 1
    error_tmp = np.inf
    while error_tmp > error:
        x_next = x - (1.0 / n + 1e-5) * f(x)
        error_tmp = abs(x - x_next)
        x = x_next
        print(x)
        n += 1
    return x


if __name__ == '__main__':
    print(solve_function(f))