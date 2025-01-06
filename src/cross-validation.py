import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from common import setup_plotting_style


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate cross-validation groups visualization')
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--columns', type=int, default=5)
    parser.add_argument('--row-spacing', type=float, default=0.2)
    parser.add_argument('--rect-height', type=float, default=0.5)
    parser.add_argument('--slice_rows', action='store_true')
    parser.add_argument('--out', type=str,
                        default='out/cross-validation.svg')

    args = parser.parse_args()
    return args


def draw_cross_validation_groups(rows, columns, row_spacing, rect_height, slice_rows):
    fig, ax = plt.subplots(figsize=(10, rows * 1.2))
    for i in range(rows):
        y_position = i * (rect_height + row_spacing)
        for j in range(columns):
            selected = j == columns - rows + i
            bg = '#e7a6a2' if selected else 'white'
            rect = patches.Rectangle(
                (j, y_position),
                1,
                rect_height,
                facecolor=bg,
                edgecolor='black'
            )
            ax.add_patch(rect)

            if selected and slice_rows:
                break

    ax.set_xlim(-0.1, columns + 0.1)
    ax.set_ylim(-0.1, rows * (rect_height + row_spacing))
    ax.set_aspect('equal')
    ax.set_axis_off()

    return fig


def main():
    args = parse_arguments()

    setup_plotting_style()
    fig = draw_cross_validation_groups(
        rows=args.rows,
        columns=args.columns,
        row_spacing=args.row_spacing,
        rect_height=args.rect_height,
        slice_rows=args.slice_rows
    )
    fig.savefig(args.out, format='svg', bbox_inches='tight')


if __name__ == '__main__':
    main()
