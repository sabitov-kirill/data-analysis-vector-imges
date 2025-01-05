import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.spatial import Voronoi, voronoi_plot_2d


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Voronoi diagram')
    parser.add_argument('--num-points', type=int, default=20)
    parser.add_argument('--min-val', type=float, default=0)
    parser.add_argument('--max-val', type=float, default=1)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--output', type=str, default='out/voronoi.svg')
    args = parser.parse_args()

    if args.min_val >= args.max_val:
        raise ValueError("min_val must be less than max_val")
    if args.num_points < 3:
        raise ValueError("num_points must be at least 3")

    return args


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.random.seed(args.seed)
    points = np.random.uniform(
        args.min_val,
        args.max_val,
        (args.num_points, 2)
    )

    vor = Voronoi(points)
    _ = voronoi_plot_2d(vor, show_vertices=False, point_size=10)
    plt.axis('off')
    plt.savefig(args.output, format='svg', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
