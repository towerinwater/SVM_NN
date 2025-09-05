import numpy as np
import cvxopt as cvx
import cvxopt.solvers
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
        self.__train()
        self.__plot()

    def __setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = self.__iris.data                # Array of shape (150, 4)
        y = self.__iris.target              # Array of shape (1,150)
        mask = (y == 0) | (y == 1)          # Taking sectosa (0) and versicolor (1)
        X = X[mask]                         # Filter rows, powerful mask method. If the position inside X has the mask in the corresponding position inside y.
        y = y[mask]                         # Same of y.
        X = X[:, [0, 1]]                    # shape (100, 2): sepal length, sepal width, the array goes down by row.
        data_max = np.max(X, axis=0)
        data_min = np.min(X, axis=0)
        return X, y, data_max, data_min

    def __train(self):
        X = self.__Data
        y = self.__labels.astype(float)  # ensure float for calculations
        y = 2 * y - 1  # map {0, 1} to {-1, +1}
        n_samples = X.shape[0]
        # Gram matrix K = X * X^T (inner products)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
        # Quadratic program setup for hard-margin SVM dual
        P = cvx.matrix(np.outer(y, y) * K)
        q = cvx.matrix(np.ones(n_samples) * -1)
        A = cvx.matrix(y, (1, n_samples))
        b = cvx.matrix(0.0)
        G = cvx.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvx.matrix(np.zeros(n_samples))
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Extract alphas
        alphas = np.ravel(solution['x'])
        # Support vectors (non-zero alphas)
        sv_mask = alphas > 1e-5
        self.alphas = alphas[sv_mask]
        self.sv_x = X[sv_mask]
        self.sv_y = y[sv_mask]
        # Compute w
        self.__w_vector = np.zeros(2)
        for n in range(len(self.alphas)):
            self.__w_vector += self.alphas[n] * self.sv_y[n] * self.sv_x[n]
        # Compute b using a support vector
        self.__b = self.sv_y[0] - np.dot(self.__w_vector, self.sv_x[0])

    def __plot(self):
        x = self.__Data[:, 0]  # sepal length
        yv = self.__Data[:, 1]  # sepal width
        y = self.__labels  # 0 or 1
        m1 = (y == 0)  # setosa
        m2 = (y == 1)  # versicolor
        plt.figure(figsize=(6, 4), dpi=130)
        plt.scatter(x[m1], yv[m1], s=45, marker='o', color='tab:orange',
                    label=self.__iris.target_names[0])
        plt.scatter(x[m2], yv[m2], s=45, marker='o', color='tab:blue',
                    label=self.__iris.target_names[1])
        # Decision boundary: w[0]*x + w[1]*y + b = 0 => y = -(w[0]/w[1])*x - b/w[1]
        xx = np.linspace(self.__min_point[0], self.__max_point[0])
        yy = -(self.__w_vector[0] / self.__w_vector[1]) * xx - (self.__b / self.__w_vector[1])
        plt.plot(xx, yy, 'k-')
        # Customize
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.title('Iris Linear Classifier')
        plt.legend()
        filename = "new.jpg"
        plt.savefig(self.__save_path + filename)
        plt.close()

def main():
    lc = LinearClassifier()
    pass

if __name__ == "__main__":
    main()