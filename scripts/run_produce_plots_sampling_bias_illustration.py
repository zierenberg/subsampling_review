import matplotlib

# reset defaults
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
# matplotlib.rcParams['font.family'] = "sans-serif"

#matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"

matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["axes.labelsize"] = 10
# matplotlib.rcParams["axes.titlesize"]= 10
matplotlib.rcParams["legend.fontsize"] = 8
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
    fig, axs = plt.subplots(1,3)

    filename="/{}/data/data_sampling_bias_illustration.npz".format(base_path)
    data = np.load(filename,'r')

    for ax in axs[1:3]:
        ax.set_xlim(1,1e5)
        ax.set_xscale("log")
        ax.set_xlabel(r"sample duration, $n$")
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0, 1.05, 1 / 10),numticks=10)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        for idx, lab in enumerate(ax.xaxis.get_ticklabels()):
            if (idx+1)%2==0:
                lab.set_visible(False)


    #var
    ax=axs[0]
    ax.set_ylabel(r"x")
    ax.set_xlabel(r"step, i")
    ax.plot(data["example"][500:900], color='slategrey')

    color_mean="#008F00"
    color_var="#1F77B4"

    #mean
    ax=axs[1]
    ax.set_ylabel(r"estimate of mean")
    #ax.set_yscale("log")
    mean_10 = np.percentile(data["mean"], 84, axis=0)
    mean_50 = np.percentile(data["mean"], 50, axis=0)
    mean_90 = np.percentile(data["mean"], 16, axis=0)
    ax.plot(data["T"], mean_50, color=color_mean)
    ax.fill_between(data["T"], mean_10, mean_90, color=color_mean, alpha=0.3, lw=0)
    #analytic solution
    #ax.axhline(0, color='black', linestyle='--')
    ts = data["T"]
    tau = 10
    correction = 2 * ( np.exp(1/tau)*( ts - 1 + np.exp(-ts/tau) ) - ts ) \
                     / np.power(np.exp(1/tau)-1, 2)\
                     / np.power(ts,2)
    #correction  = (np.exp(-(ts-1)/tau)-1)/(1-np.exp(1/tau))/ts
    ax.plot(ts, np.sqrt(1/ts+correction), color='black', linestyle=':')
    ax.plot(ts, -np.sqrt(1/ts+correction), color='black', linestyle=':')

    #var
    ax=axs[2]
    ax.set_ylabel(r"estimate of variance")
    var_10 = np.percentile(data["var"], 84, axis=0)
    var_50 = np.percentile(data["var"], 50, axis=0)
    var_90 = np.percentile(data["var"], 16, axis=0)
    ax.plot(data["T"], var_50, color=color_var)
    ax.fill_between(data["T"], var_10, var_90, color=color_var, alpha=0.3, lw=0)
    #analytic solution
    ax.axhline(1, color='black', linestyle='--')
    ax.plot(ts, 1-1/ts-correction, color='black', linestyle=':')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig('/{}/plots/sampling_bias_illustration.pdf'.format(base_path), bbox_inches='tight')



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
