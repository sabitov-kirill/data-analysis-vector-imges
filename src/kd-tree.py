import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate and visualize 2D KD-tree')
    parser.add_argument('--n-points', type=int, default=5)
    parser.add_argument('--min-val', type=float, default=0.0)
    parser.add_argument('--max-val', type=float, default=5.0)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--test-x', type=float, default=1)
    parser.add_argument('--test-y', type=float, default=4)
    parser.add_argument('--seed', type=int, default=11001)
    parser.add_argument('--max_draw_depth', type=int, default=10)
    parser.add_argument('--figsize', type=int, default=5)
    parser.add_argument('--output', default='out/kdtree.svg')
    return parser.parse_args()


def plot_kdtree(points, tree, x_bounds, y_bounds, max_draw_depth, depth=0):
    if depth >= max_draw_depth:
        return

    axis = depth % 2
    split_point = np.median(points[:, axis])

    plt.plot(
        [split_point, split_point] if axis == 0 else x_bounds,
        y_bounds if axis == 0 else [split_point, split_point],
        'k-' if axis == 0 else 'r-',
        zorder=1 if axis == 0 else 0
    )

    mask = points[:, axis] < split_point
    new_bounds = {
        'left': (x_bounds[0], split_point) if axis == 0 else (y_bounds[0], split_point),
        'right': (split_point, x_bounds[1]) if axis == 0 else (split_point, y_bounds[1])
    }

    for side, curr_mask in [('left', mask), ('right', ~mask)]:
        if np.any(curr_mask):
            new_bound = new_bounds[side]
            next_x_bounds = new_bound if axis == 0 else x_bounds
            next_y_bounds = new_bound if axis == 1 else y_bounds
            plot_kdtree(points[curr_mask], tree, next_x_bounds, next_y_bounds,
                       max_draw_depth, depth + 1)


def visualize_neighbors(tree, test_point, k):
    distances, indices = tree.query([test_point], k=k)
    indices = indices.flatten()
    distances = distances.flatten()
    neighbors = np.array([tree.data[i] for i in indices])
    
    # Neighbors
    plt.scatter(neighbors[:, 0], neighbors[:, 1], c='purple', s=100, zorder=3)
    
    # Distance
    plt.gca().add_patch(plt.Circle(test_point, distances[-1], 
                                  fill=False, color='purple', ls='--', zorder=1))
    plt.plot([test_point[0], neighbors[-1, 0]], 
            [test_point[1], neighbors[-1, 1]], ':', c='purple', zorder=2)
    
    # Test point
    plt.scatter(*test_point, c='red', marker='*', s=200, zorder=4, 
               edgecolors='black', lw=0.5)


def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    points = np.random.uniform(
        low=args.min_val, high=args.max_val, size=(args.n_points, 2))
    test_point = np.array([args.test_x, args.test_y])
    tree = KDTree(points)

    padding = (args.max_val - args.min_val) * 0.05
    limits = (args.min_val - padding, args.max_val + padding)
    plt.figure(figsize=(args.figsize, args.figsize))

    plot_kdtree(points, tree, x_bounds=(args.min_val, args.max_val),
                y_bounds=(args.min_val, args.max_val), max_draw_depth=args.max_draw_depth)
    visualize_neighbors(tree, test_point, args.k)

    plt.xlim(*limits)
    plt.ylim(*limits)
    plt.scatter(points[:, 0], points[:, 1], c='black', s=100, zorder=2)
    plt.axis('off')
    plt.savefig(args.output, format='svg', bbox_inches='tight')


if __name__ == '__main__':
    main()
