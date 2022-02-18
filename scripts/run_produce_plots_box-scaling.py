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
matplotlib.rcParams["figure.figsize"] = [3.4, 2.6]  # APS single column
matplotlib.rcParams["figure.dpi"] = 300  # this primarily affects the size on screen

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator
import os
import scipy.optimize

base_path = os.path.join(*os.path.normpath(__file__).split(os.path.sep)[:-2])

def main():
    def string(l):
        return "l={:d}".format(l)
    def func_exp(x,A,tau,O):
        return A * np.exp(-x/tau) + O

    # figure scaling of zero crossing
    fig, ax = plt.subplots(1,1)
    # ax[0]: scaling
    ax.set_ylabel(r"characteristic length $r^\ast$")
    ax.set_xlabel(r"Window size $W$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _fix_log_ticks(ax.xaxis, every=1)
    _fix_log_ticks(ax.yaxis, every=2)
    #ax.set_ylim(1e-8,0.3)

    x=np.arange(10,512)
    ax.plot(x,0.30*x,'--', color='black', linewidth=0.4)

    colors=['orange', 'yellowgreen', 'green']

    h=1e-7
    ms=[1.09, 1.10, 1.2]
    for i,m in enumerate(ms):
        try:
            filename="/{}/data/data_box_scaling_m{:06.4f}_h{:.2e}.npz".format(base_path,m,h)
            f = np.load(filename,'r')

            #fig = plt.figure()
            ls = f["l"]
            rs = []
            for l in ls:
                # plot correlation function for intermediate check
                x = f["correlation/window_lag/{}".format(string(l))]
                y = f["correlation/window/{}".format(string(l))]
                #plt.plot(x,y)

                # correlation length scale
                try:
                    # find first negative correlation
                    r = x[next(x for x,c in enumerate(y) if c <0)]
                    # find from fit to exponential
                    #p0 = (1, 1, -1) # start with values near those we expect
                    #params, cv = scipy.optimize.curve_fit(func_exp, x, y, p0)
                    #A, tau, O = params
                    #r = -tau*np.log(-O/A)
                except:
                    r = 0
                rs.append(r)
            #plt.show()

            # plot characteristic lengths
            ax.plot(ls, rs, '.-', label='m={:05.3f}'.format(m), zorder=len(ms)-i, color=colors[i])
        except:
            print("did not find ", m)
            pass

    #ax.legend(loc='upper left', bbox_to_anchor=(0.01,0.01))
    ax.legend(loc='upper left')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig('/{}/plots/box_scaling.pdf'.format(base_path))


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
