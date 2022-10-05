import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').to_numpy()
n = data.shape[0]
x_init = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x_init, y)
plt.xlabel('Area (square metre)')
plt.ylabel('Price (dollars)')

# add 1's in the 0th column of x
x = np.zeros((x_init.shape[0], x_init.shape[1] + 1), dtype=x_init.dtype)
x[:, 0] = 1
x[:, 1:] = x_init

w = np.array([0., 1.]).reshape(-1, 1)

alpha = 1e-5
max_iter = 100
for i in range(1, max_iter):
    diff = np.dot(x, w) - y
    cost = 0.5 * np.sum(diff * diff)
    w[0] -= alpha * np.sum(diff)
    w[1] -= alpha * np.sum(np.multiply(diff, x[:, 1].reshape(-1, 1)))
    print(cost)

pred = np.dot(x, w)
plt.plot((x[0][1], x[n-1][1]), (pred[0], pred[n-1]), 'r')
plt.show()

area = [50, 100, 150]
for a in area:
    print(f'Price of {a} m^2 house is:', round((w[0] + w[1] * a).item(), 3))
