import numpy as np
import matplotlib.pyplot as plt
import argparse

from common import save_plt, setup_plotting_style


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=25)
    parser.add_argument('--mean', nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument('--cov', nargs=4, type=float,
                        default=[0.7, 0.45, 0.45, 0.5])
    parser.add_argument('--out', default='out/pca.svg')
    parser.add_argument('--seed', type=int, default=30)
    return parser.parse_args()


def generate_data(n_points, mean, cov_matrix):
    return np.random.multivariate_normal(mean, cov_matrix, n_points)


def calculate_pca(data):
    data_centered = data - np.mean(data, axis=0)
    _, _, Vt = np.linalg.svd(data_centered, full_matrices=False)
    return Vt[0]


def point_to_line_distance(point, line_point, line_direction):
    vec = point - line_point
    projection = np.dot(vec, line_direction) * line_direction
    perpendicular = vec - projection
    return perpendicular


def plot_pca(data, principal_direction, data_mean):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Data points')
    line_points = np.vstack(
        [data_mean + 3*principal_direction, data_mean - 3*principal_direction])
    plt.plot(line_points[:, 0], line_points[:, 1], 'r-', label='PCA line')
    for point in data:
        perp = point_to_line_distance(point, data_mean, principal_direction)
        end_point = point - perp
        plt.plot([point[0], end_point[0]], [
                 point[1], end_point[1]], 'g--', alpha=0.3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.axis('equal')


def main():
    args = parse_arguments()
    if args.seed is not None:
        np.random.seed(args.seed)

    cov_matrix = np.array(args.cov).reshape(2, 2)
    data = generate_data(args.n_points, args.mean, cov_matrix)

    principal_direction = calculate_pca(data)
    data_mean = np.mean(data, axis=0)

    setup_plotting_style()
    plot_pca(data, principal_direction, data_mean)
    save_plt(args.out)


if __name__ == "__main__":
    main()
