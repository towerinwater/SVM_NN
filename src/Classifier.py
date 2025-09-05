import numpy as np
import cvxopt as cvx
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os

class LinearClassifier:
    def __init__(self):
        self.__iris = load_iris()
        self.__current_path = os.path.dirname(os.path.abspath(__file__))
        self.__save_path = os.path.join(self.__current_path, "../graph/")
        os.makedirs(self.__save_path, exist_ok=True)
        self.__Data, self.__labels, self.__max_point, self.__min_point = self.__setup()
        self.__w_vector = 
        self.__plot()
        pass

    def __setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = self.__iris.data          # shape (150, 4)
        y = self.__iris.target        # shape (150,)

        mask = (y == 0) | (y == 1)    # boolean mask of length 150
        X = X[mask]                   # filter rows
        y = y[mask]

        X = X[:, [0, 1]]              # shape (~100, 2)

        data_max = np.ndarray((1,2), dtype = float)
        data_min = np.ndarray((1,2), dtype = float)

        data_max[0] = max(X[:, 0])
        data_max[1] = max(X[:, 1])

        data_min[0] = min(X[:, 0])
        data_min[1] = min(X[:, 1])

        return X, y, data_max, data_min

    def __plot(self):
        x = self.__Data[:, 0]          # sepal width
        yv = self.__Data[:, 1]         # sepal length
        y  = self.__labels                  # labels (1=versicolor, 2=virginica)

        m1 = (y == 0)                  # class 1 mask
        m2 = (y == 1)                  # class 2 mask

        plt.figure(figsize=(6,4), dpi=130)
        plt.scatter(x[m1], yv[m1], s=45, marker='o', color='tab:orange',
                    label=self.__labels[0])   # versicolor
        plt.scatter(x[m2], yv[m2], s=45, marker='o', color='tab:blue',
                    label=self.__labels[1])   # virginica

        # Customize
        plt.xlabel('Sepal Width')
        plt.ylabel('Sepal Length')
        plt.title('Iris')

        filename = "test.jpg"
        plt.savefig(self.__save_path + filename)
    
    # def __linear(self):

    