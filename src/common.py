from matplotlib import pyplot as plt, rc


def setup_plotting_style():
    rc('font', family='serif', serif=['Computer Modern'], size=13)
    rc('text', usetex=True)


def save_plt(path):
    plt.tight_layout()
    plt.savefig(path, format='svg', bbox_inches='tight')
    plt.close()
