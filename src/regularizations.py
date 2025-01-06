# Reference: https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf
# Section 3.4.1 for Ridge, 3.4.2 for Lassos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip
import argparse

from common import save_plt, setup_plotting_style


def load_and_prepare_data(url):
    df = pd.read_csv(url, sep="\\s+")
    X = df.drop('lpsa', axis=1)
    y = df['lpsa']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled


def calculate_regularization_paths(X_scaled, y, alphas):
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    U, Vt = svd_flip(U, Vt)

    ridge_coefs, lasso_coefs, ridge_dof = [], [], []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        ridge_coefs.append(ridge.coef_)
        ridge_dof.append(np.sum(s**2 / (s**2 + alpha)))

        lasso = Lasso(alpha=alpha)
        lasso.fit(X_scaled, y)
        lasso_coefs.append(lasso.coef_)

    return np.array(ridge_coefs), np.array(lasso_coefs), ridge_dof


def create_plots(X, ridge_coefs, lasso_coefs, ridge_dof):
    plt.figure(figsize=(15, 6))

    lasso_norms = np.sum(np.abs(lasso_coefs), axis=1)
    lasso_x = lasso_norms/np.max(lasso_norms)

    n_columns = len(X.columns)
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    for i in range(n_columns):
        plt.plot(ridge_dof, ridge_coefs[:, i], label=X.columns[i],
                 marker='o', markersize=3)
    plt.xlabel(
        'Degrees of Freedom $df(\\lambda) = \\mathrm{tr}(X(X^T X + \\lambda I)^{-1}X^T)$')
    plt.ylabel('Coefficient values')
    plt.title('Ridge Regularization Path')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    for i in range(n_columns):
        plt.plot(lasso_x, lasso_coefs[:, i], label=X.columns[i],
                 marker='o', markersize=3)
    plt.xlabel('Shrinking Factor $s = \\frac{\\kappa}{\\max_i|\\beta_i|}$')
    plt.ylabel('Coefficient values')
    plt.title('Lasso Regularization Path')
    plt.legend(loc='upper right')


def main():
    parser = argparse.ArgumentParser(
        description='Generate regularization path plots')
    parser.add_argument('--out', default='out/regularizations.svg')
    args = parser.parse_args()

    data_url = "http://www.stat.cmu.edu/~ryantibs/statcomp-S18/data/pros.dat"
    alphas = np.logspace(-3, 6, 100)

    X, y, X_scaled = load_and_prepare_data(data_url)
    ridge_coefs, lasso_coefs, ridge_dof = calculate_regularization_paths(
        X_scaled, y, alphas)

    setup_plotting_style()
    create_plots(X, ridge_coefs, lasso_coefs, ridge_dof)
    save_plt(args.out)


if __name__ == "__main__":
    main()
