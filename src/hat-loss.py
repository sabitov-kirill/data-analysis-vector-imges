import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot hat loss function')
    parser.add_argument('--output', type=str, default='out/hat-loss.svg')
    parser.add_argument('--figsize', type=float, nargs=2, default=(4, 4))
    parser.add_argument('--xlim', type=float, nargs=2, default=(-2, 2))
    parser.add_argument('--ylim', type=float, nargs=2, default=(0, 3))
    return parser.parse_args()


def hat_loss(x):
    return np.maximum(0, 1 - np.abs(x))


def main():
    args = parse_arguments()

    x = np.linspace(args.xlim[0], args.xlim[1], 1000)
    y = hat_loss(x)

    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=args.figsize)
    plt.plot(x, y, 'b-', linewidth=2, label='$(1-|f(x_i)|)_+$')

    plt.grid(True, alpha=0.3)
    plt.xlabel('$f(x_i)$')
    plt.xlim(args.xlim)
    plt.ylim(args.ylim)
    plt.grid(False)
    plt.legend()

    plt.savefig(args.output, format='svg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
