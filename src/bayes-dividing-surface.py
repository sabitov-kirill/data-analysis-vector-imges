import argparse
import numpy as np
import plotly.graph_objects as go
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det
from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
from skimage import measure


@dataclass
class DataConfig:
    n_samples: int
    mean1: np.ndarray
    mean2: np.ndarray
    cov1: np.ndarray
    cov2: np.ndarray


class DataGenerator:
    def __init__(self, config: DataConfig):
        self.config = config

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        X1 = np.random.multivariate_normal(
            self.config.mean1, self.config.cov1, self.config.n_samples)
        X2 = np.random.multivariate_normal(
            self.config.mean2, self.config.cov2, self.config.n_samples)
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(self.config.n_samples),
                      np.ones(self.config.n_samples)])
        return X, y


@dataclass
class GridConfig:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    n_points: int


class BayesClassifier:
    def __init__(self, mean1, mean2, cov1, cov2):
        self.mean1, self.mean2 = mean1, mean2
        self.cov1, self.cov2 = cov1, cov2
        self.inv_cov1, self.inv_cov2 = inv(cov1), inv(cov2)

    def compute_densities(self, X_grid, Y_grid):
        pos = np.dstack((X_grid, Y_grid))
        rv1 = multivariate_normal(self.mean1, self.cov1)
        rv2 = multivariate_normal(self.mean2, self.cov2)
        return rv1.pdf(pos), rv2.pdf(pos)

    def compute_decision_boundary(self, X_grid, Y_grid):
        Z_boundary = np.zeros_like(X_grid)
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                point = np.array([X_grid[i, j], Y_grid[i, j]])
                Z_boundary[i, j] = self._compute_log_likelihood_ratio(point)
        return Z_boundary

    def _compute_log_likelihood_ratio(self, point):
        diff1, diff2 = point - self.mean1, point - self.mean2
        return (-0.5 * (diff1.T @ self.inv_cov1 @ diff1 - diff2.T @ self.inv_cov2 @ diff2) +
                0.5 * np.log(det(self.cov2)/det(self.cov1)))


@dataclass 
class VisualizerConfig:
    width: int = 1500
    height: int = 700
    eye: Tuple[float, float, float] = (1.5, -1.5, 0.25)
    z_range: Tuple[float, float] = (-0.1, 0.2)
    x_range: Tuple[float, float] = (-2, 4)
    y_range: Tuple[float, float] = (-2, 4)


class Visualizer(ABC):
    def __init__(self, config: VisualizerConfig):
        self.config = config

    @abstractmethod
    def create_visualization(self, X_grid, Y_grid, Z1, Z2, boundary_data) -> None:
        pass

    @abstractmethod
    def save(path) -> None:
        pass

    @abstractmethod
    def show():
        pass


class PlotlyVisualizer(Visualizer):
    def create_visualization(self, X_grid, Y_grid, Z1, Z2, boundary_data) -> 'PlotlyVisualizer':
        self.fig = go.Figure()
        self._add_surfaces(self.fig, X_grid, Y_grid, Z1, Z2)
        self._add_decision_boundary(self.fig, boundary_data)
        self._update_layout(self.fig)
        return self

    def save(self, path: str) -> 'PlotlyVisualizer':
        if self.fig is None:
            raise RuntimeError("Create visualization first")

        if path.endswith('.html'):
            self.fig.write_html(path)
        else:
            self.fig.write_image(path)
        return self

    def show(self) -> 'PlotlyVisualizer':
        if self.fig is None:
            raise RuntimeError("Create visualization first")

        self.fig.show()
        return self

    def _add_surfaces(self, fig, X_grid, Y_grid, Z1, Z2):
        for Z, colorscale, color in [(Z1, 'Blues', 'blue'), (Z2, 'Reds', 'red')]:
            start = np.percentile(Z, 90)
            end = np.max(Z)
            size = (end - start) / 3
            fig.add_trace(
                go.Surface(
                    x=X_grid, y=Y_grid, z=Z,
                    colorscale=colorscale,
                    showscale=False,
                    contours={
                        "z": {"show": True, "color": color, "project_z": True, "start": start, "end": end, "size": size}}
                )
            )

    def _add_decision_boundary(self, fig, boundary_data):
        x_boundary, y_boundary, z_boundary = boundary_data
        fig.add_trace(
            go.Scatter3d(
                x=x_boundary,
                y=y_boundary,
                z=np.full_like(z_boundary, self.config.z_range[0]),
                mode='lines',
                line=dict(color='green', width=5),
                name='Boundary Projection',
                showlegend=False
            )
        )

    def _update_layout(self, fig):
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            scene=dict(
                xaxis=dict(
                    title='X',
                    range=self.config.x_range
                ),
                yaxis=dict(
                    title='Y', 
                    range=self.config.y_range
                ),
                zaxis=dict(
                    title='Probability Density',
                    range=self.config.z_range
                ),
                camera=dict(
                    eye=dict(
                        x=self.config.eye[0],
                        y=self.config.eye[1],
                        z=self.config.eye[2]
                    )
                ),
                aspectmode='cube'
            ),
            showlegend=True
        )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True,
                                       help='Visualization mode')

    show_parser = subparsers.add_parser('show', help='Display the plot')
    add_common_args(show_parser)

    save_parser = subparsers.add_parser('save', help='Save the plot to file')
    save_parser.add_argument('--out', type=str, default='out/bayes.svg',
                             help='Output file path (.html or image format)')
    add_common_args(save_parser)

    return parser.parse_args()


def add_common_args(parser):
    data_group = parser.add_argument_group('Data Generation')
    data_group.add_argument('--n_samples', type=int, default=1000)
    data_group.add_argument('--mean1', type=float, nargs=2, default=[0, 0],
                            help='2D mean vector for first distribution')
    data_group.add_argument('--mean2', type=float, nargs=2, default=[2, 2],
                            help='2D mean vector for second distribution')
    data_group.add_argument('--cov1', type=float, nargs=4,
                            default=[1, 0, 0, 1],
                            help='2x2 covariance matrix for first distribution (flattened)')
    data_group.add_argument('--cov2', type=float, nargs=4,
                            default=[2, 1.5, 1.5, 2],
                            help='2x2 covariance matrix for second distribution (flattened)')

    grid_group = parser.add_argument_group('Grid')
    grid_group.add_argument('--grid_points', type=int, default=100)

    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--viz_dims', type=int, nargs=2,
                           default=[700, 700],
                           help='Width and height of the plot')
    viz_group.add_argument('--eye', type=float, nargs=3,
                           default=[1.5, -1.5, 0.25],
                           help='Camera position (x, y, z)')
    viz_group.add_argument('--z_limits', type=float, nargs=2,
                           default=[-0.1, 0.2],
                           help='Z-axis limits (min, max)')
    viz_group.add_argument('--xlim', type=float, nargs=2,
                       default=[-2, 4],
                       help='X-axis limits (min, max)')
    viz_group.add_argument('--ylim', type=float, nargs=2,
                        default=[-2, 4], 
                        help='Y-axis limits (min, max)')
    viz_group.add_argument('--margin', type=float,
                        default=0.05,
                        help='Figure margins as fraction of axis range')



def main():
    args = parse_args()

    data_config = DataConfig(
        n_samples=args.n_samples,
        mean1=np.array(args.mean1),
        mean2=np.array(args.mean2),
        cov1=np.array(args.cov1).reshape(2, 2),
        cov2=np.array(args.cov2).reshape(2, 2)
    )

    x = np.linspace(-2, 4, args.grid_points)
    y = np.linspace(-2, 4, args.grid_points)
    X_grid, Y_grid = np.meshgrid(x, y)

    viz_config = VisualizerConfig(
        width=args.viz_dims[0],
        height=args.viz_dims[1], 
        eye=tuple(args.eye),
        z_range=tuple(args.z_limits),
        x_range=tuple(args.xlim),
        y_range=tuple(args.ylim)
    )

    classifier = BayesClassifier(
        data_config.mean1, data_config.mean2,
        data_config.cov1, data_config.cov2
    )

    Z1, Z2 = classifier.compute_densities(X_grid, Y_grid)
    Z_boundary = classifier.compute_decision_boundary(X_grid, Y_grid)
    contours = measure.find_contours(Z_boundary, 1e-6)
    boundary_points = contours[0]
    x_coords = x[boundary_points[:, 1].astype(int)]
    y_coords = y[boundary_points[:, 0].astype(int)]
    z_coords = np.zeros_like(x_coords)

    visualizer = PlotlyVisualizer(viz_config)
    visualization = visualizer.create_visualization(
        X_grid, Y_grid, Z1, Z2,
        (x_coords, y_coords, z_coords)
    )

    if args.mode == 'show':
        visualization.show()
    else:
        visualization.save(args.out)


if __name__ == "__main__":
    main()
