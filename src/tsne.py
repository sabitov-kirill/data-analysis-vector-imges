import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import t, norm

from common import save_plt, setup_plotting_style


def plot_distributions(figsize=(10, 6)):
    x = np.linspace(-4, 4, 1000)
    p_dist = norm.pdf(x, loc=0, scale=1)
    q_dist = t.pdf(x, df=1, loc=0, scale=1)

    plt.figure(figsize=figsize)
    plt.plot(x, p_dist, 'b-', label='Gaussian, $N(\\alpha=0, \\sigma^2=1)$ (p)')
    plt.plot(x, q_dist, 'r-', label='Student-t, $t(n=1)$ (q)')
    plt.ylim(0)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot t-SNE distributions')
    parser.add_argument('--out', default='out/tsne-distributions.svg')
    parser.add_argument('--figsize', type=int, nargs=2, default=(10, 6))
    args = parser.parse_args()

    setup_plotting_style()
    plot_distributions(figsize=args.figsize)
    save_plt(args.out)
