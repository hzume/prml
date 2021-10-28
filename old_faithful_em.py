from em_algorithm import EM_mixed_gauss, Gaussian
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv("old_faithful.csv", header=None)
x = df[1].values
y = df[2].values

x = (x - np.mean(x))/ np.std(x)
y = (y - np.mean(y)) / np.std(y)

data = np.stack([x,y]).T

gamma = EM_mixed_gauss(2, data, 20) # [N, K]
x = df[1].values
y = df[2].values

cmap = ListedColormap(["red", "blue"])
N = len(x)
c = np.array([(gamma[n][0] > gamma[n][1]) for n in range(N)])
plt.scatter(x=x, y=y, c=c, cmap=cmap)
plt.show()
