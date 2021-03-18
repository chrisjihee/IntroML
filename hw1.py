from sklearn import datasets
from sklearn.utils import Bunch
import numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    iris: Bunch = datasets.load_iris()
    print("-" * 120)
    print(f"filename={iris['filename']}")
    print("-" * 120)
    print(f"feature_names={iris['feature_names']}")
    print("-" * 120)
    print(f"target_names={iris['target_names']}")
    print("-" * 120)
    print(f"data={iris['data']}")
    print("-" * 120)
    print(f"target={iris['target']}")
    print("-" * 120)
    X: np.ndarray = iris['data']
    y: np.ndarray = iris['target']

    # Mean Vector (with Numpy)
    mean_vector = X.mean(axis=0)
    print(f"Mean Vector: [{', '.join('%.2f' % x for x in mean_vector)}]")

    # Mean Vector (without Numpy)
    sum_vector = [0.0, 0.0, 0.0, 0.0]
    for x in X:
        sum_vector[0] += x[0]
        sum_vector[1] += x[1]
        sum_vector[2] += x[2]
        sum_vector[3] += x[3]
    mean_vector = [x / len(X) for x in sum_vector]
    print(f"Mean Vector: [{', '.join('%.2f' % x for x in mean_vector)}]")

    # print(np.cov(X))


if __name__ == '__main__':
    print_hi('PyCharm')
