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
matplotlib.rcParams["figure.figsize"] = [6.8, 2.2]  # APS double column
matplotlib.rcParams["figure.dpi"] = 300  # this primarily affects the size on screen

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator

def main():
    fig, axs = plt.subplots(1,2)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        _fix_log_ticks(ax.xaxis, every=1)
        _fix_log_ticks(ax.yaxis, every=2)
        ax.set_ylabel(r"distribution $P(s)$")
        ax.set_xlabel(r"avalanche size $s$")
        ax.set_ylim(1e-8,0.3)

    #filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.644_seed=1000.h5"
    #filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.644_ref=4_seed=1000.h5"
    #filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.8_ref=4_seed=1000.h5"
    #filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.9_ref=4_seed=1000.h5"
    #filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.35_ref=4_seed=1000.h5"
    filename="../data/processed/processed_L=128_num_avalanches=100000_p_rec=0.4_ref=4_seed=1000.h5"
    #filename="../data/processed/processed_L=1024_num_avalanches=100000_p_rec=0.644_seed=1000.h5"
    f = h5py.File(filename,'r')

    # random subsampling
    ax = axs[0]
    plot_data = np.transpose(f["size"]["full"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color='black', label=r"$100\%$")

    plot_data = np.transpose(f["size"]["random"]["64"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color= _alpha_to_solid_on_bg('blue', 1.0), label=r"$\left(\frac{64}{128}\right)=50\%$")

    plot_data = np.transpose(f["size"]["random"]["16"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color= _alpha_to_solid_on_bg('blue', 0.7), label=r"$\left(\frac{16}{128}\right)=25\%$")

    plot_data = np.transpose(f["size"]["random"]["4"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color= _alpha_to_solid_on_bg('blue', 0.3), label=r"$\left(\frac{4}{128}\right)=12.5\%$")

    ax.legend(loc='lower left', bbox_to_anchor=(0.01,0.01))

    # window subsampling
    ax = axs[1]
    plot_data = np.transpose(f["size"]["full"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color='black', label=r"$128$")

    plot_data = np.transpose(f["size"]["window"]["64"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color=_alpha_to_solid_on_bg('orange', 1.0), label=r"$64$")

    plot_data = np.transpose(f["size"]["window"]["16"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color=_alpha_to_solid_on_bg('orange', 0.7), label=r"$16$")

    plot_data = np.transpose(f["size"]["window"]["4"])
    ax.plot(plot_data[:,0], plot_data[:,1], '-', color=_alpha_to_solid_on_bg('orange', 0.3), label=r"$4$")

    ax.legend(loc='lower left', bbox_to_anchor=(0.01,0.01))

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig('../plots/bn_subsampling.pdf')


def _fix_log_ticks(ax_el, every=1, hide_label_condition=lambda idx: False):
    """
        this can adapt log ticks to only show every second tick, or so.

        # Parameters
        ax_el: usually `ax.yaxis`
        every: 1 or 2
        hide_label_condition : function e.g. `lambda idx: idx % 2 == 0`
    """
    ax_el.set_major_locator(LogLocator(base=10, numticks=10))
    ax_el.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(0, 1.05, every / 10), numticks=10)
    )
    ax_el.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for idx, lab in enumerate(ax_el.get_ticklabels()):
        if hide_label_condition(idx):
            lab.set_visible(False)


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
