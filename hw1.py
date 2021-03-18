from sklearn import datasets
from sklearn.utils import Bunch
import numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    data: Bunch = datasets.load_iris()
    print("-" * 120)
    print(f"filename={data['filename']}")
    print("-" * 120)
    print(f"feature_names={data['feature_names']}")
    print("-" * 120)
    print(f"target_names={data['target_names']}")
    print("-" * 120)
    print(f"data={data['data']}")
    print("-" * 120)
    print(f"target={data['target']}")
    print("-" * 120)
    print(f"DESCR={data['DESCR']}")
    print("-" * 120)


if __name__ == '__main__':
    print_hi('PyCharm')
