import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
save_path = current_path + "/../graph"

data = load_iris()
X = data.data[:, :2]
y = data.target
y = np.where(y == 0, -1, 1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset (Setosa vs. Non-Setosa)')

my_file = "output.jpg"
plt.savefig(os.path.join(save_path, my_file))