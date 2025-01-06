import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import argparse

from common import save_plt, setup_plotting_style


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Decision tree depth visualization')
    parser.add_argument('--out', default='out/decision-tree-depth.svg')
    parser.add_argument('--xlim', type=float, nargs=2, default=(-4, 3))
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--n-outliers', type=int, default=15)
    parser.add_argument('--outliers-lim', type=float, nargs=2, default=(-5, 5))
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    X = np.linspace(*args.xlim, 150).reshape(-1, 1)
    y = ((X.ravel()/3) - 3)**2 + ((X.ravel()/4)**3)*10 - 10

    np.random.seed(args.seed)
    y += np.random.normal(0, 0.1, y.shape)
    keep_mask = np.random.choice([True, False], size=len(y), p=[0.8, 0.2])
    X = X[keep_mask]
    y = y[keep_mask]
    outliers = np.random.choice(len(y), args.n_outliers, replace=False)
    y[outliers] += np.random.uniform(*args.outliers_lim, args.n_outliers)

    setup_plotting_style()
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, color='black', s=10, label='Data')

    colors = ['green', 'red', 'blue', 'purple']
    X_test = np.linspace(*args.xlim, 300).reshape(-1, 1)
    for depth, color in zip(args.depths, colors):
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X, y)
        y_pred = regr.predict(X_test)
        plt.plot(X_test, y_pred, color=color,
                 label=f'depth={depth}', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    save_plt(args.out)


if __name__ == "__main__":
    main()
