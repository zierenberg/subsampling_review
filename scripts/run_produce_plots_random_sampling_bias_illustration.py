import matplotlib

# reset defaults
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"

#matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"

matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
# matplotlib.rcParams["axes.titlesize"]= 10
matplotlib.rcParams["legend.fontsize"] = 6
# matplotlib.rcParams["legend.title_fontsize"] = 8

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False

# avoid using `plt.subplots(figsize=(3.4, 2.7))` every time
matplotlib.rcParams["figure.figsize"] = [6.8, 2.4]  # APS double column
matplotlib.rcParams["figure.dpi"] = 300  # this primarily affects the size on screen

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, LogLocator
import os

base_path = os.path.join(*os.path.normpath(__file__).split(os.path.sep)[:-2])

def main():
    fig, axs = plt.subplots(1,2)

    filename="/{}/data/data_random_sampling_bias_illustration.npz".format(base_path)
    data = np.load(filename,'r')

    for ax in axs[0:2]:
        ax.set_xscale("log")
        ax.set_xlabel(r"number of correlated samples $n$")

    #mean
    ax=axs[0]
    ax.set_ylabel(r"estimate of mean")
    mean_10 = np.percentile(data["mean"], 25, axis=0)
    mean_50 = np.percentile(data["mean"], 50, axis=0)
    mean_90 = np.percentile(data["mean"], 75, axis=0)
    ax.plot(data["T"], mean_50, color='darkgreen')
    ax.fill_between(data["T"], mean_10, mean_90, color='darkgreen', alpha=0.3, lw=0)
    ax.axhline(0, color='black', linestyle='--')

    #var
    ax=axs[1]
    ax.set_ylabel(r"estimate of variance")
    var_10 = np.percentile(data["var"], 25, axis=0)
    var_50 = np.percentile(data["var"], 50, axis=0)
    var_90 = np.percentile(data["var"], 75, axis=0)
    ax.plot(data["T"], var_50, color='firebrick')
    ax.fill_between(data["T"], var_10, var_90, color='firebrick', alpha=0.3, lw=0)
    ax.axhline(1, color='black', linestyle='--')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig('/{}/plots/sampling_random_bias_illustration.pdf'.format(base_path))



def _alpha_to_solid_on_bg(base, alpha, bg="white"):
    """
        Probide a color to start from `base`, and give it opacity `alpha` on
        the background color `bg`
    """

    def rgba_to_rgb(c, bg):
        bg = matplotlib.colors.to_rgb(bg)
        alpha = c[-1]

        res = (
            (1 - alpha) * bg[0] + alpha * c[0],
            (1 - alpha) * bg[1] + alpha * c[1],
            (1 - alpha) * bg[2] + alpha * c[2],
        )
        return res

    new_base = list(matplotlib.colors.to_rgba(base))
    new_base[3] = alpha
    return matplotlib.colors.to_hex(rgba_to_rgb(new_base, bg))

if __name__ == "__main__":
    main()
