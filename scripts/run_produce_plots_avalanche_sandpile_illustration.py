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
matplotlib.rcParams["figure.figsize"] = [9.0, 4.4]  # APS double column
matplotlib.rcParams["figure.dpi"] = 300  # this primarily affects the size on screen

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, LogLocator
import os

base_path = os.path.join(*os.path.normpath(__file__).split(os.path.sep)[:-2])

def main():
    fig, axs = plt.subplots(2,4)
    for ax in axs[0]:
        ax.set_ylabel(r"projected activity $a_y = \sum_x a_{xy}$")
        ax.set_xlabel(r"time")


    filename="/{}/data/data_avalanche_sandpile_illustration.npz".format(base_path)
    data = np.load(filename,'r')
    print(data)

    #cmap = plt.get_cmap('plasma').copy()
    #cmap.set_under('white')
    cmap='Greys'

    max_num = np.matrix(data["time/full"]).max()
    print(max_num)

    ax = axs[0,0]
    ax.set_title('full (32x32)')
    ax.imshow(data["time/full"], vmin=0, vmax=max_num, cmap=cmap,  interpolation='nearest', aspect='auto')

    ax = axs[0,1]
    ax.set_title('random(64)')
    pos = ax.imshow(data["time/rand"], vmin=0, vmax=max_num, cmap=cmap, interpolation='nearest', aspect='auto')

    ax = axs[0,2]
    ax.set_title('window(8x8)')
    #sample_range = data["wind"]
    #sample_range[:] = 0
    #sample_range[24:40,:] = 0.1
    #ax.imshow(sample_range, cmap='Greys',  interpolation='nearest', aspect='auto', alpha=0.4)
    pos = ax.imshow(data["time/wind"], vmin=0, vmax=max_num, cmap=cmap,  interpolation='nearest', aspect='auto')
    ax.axhline(11.5, color=_alpha_to_solid_on_bg('orange', 1.0), linewidth=1.0)
    ax.axhline(19.5, color=_alpha_to_solid_on_bg('orange', 1.0), linewidth=1.0)


    for ax in axs[0]:
        ax.invert_yaxis()

    #######################
    #### spatial avalanches
    ax = axs[1,0]
    ax.imshow(data["space/full"].transpose(), vmin=0, vmax=max_num,cmap=cmap, interpolation='nearest', aspect='auto')

    ax = axs[1,1]
    ax.imshow(data["space/rand"].transpose(), vmin=0, vmax=max_num,cmap=cmap, interpolation='nearest', aspect='auto')

    ax = axs[1,2]
    ax.imshow(data["space/wind"].transpose(), vmin=0, vmax=max_num,cmap=cmap, interpolation='nearest', aspect='auto')


    layer_rand = data["layer/rand"]
    layer_wind = data["layer/wind"]
    for ax in axs[1]:
        ax.set_ylabel(r'y')
        ax.set_xlabel(r'x')

        rect = patches.Rectangle((11.5, 11.5), 8, 8, linewidth=1.0, edgecolor=_alpha_to_solid_on_bg('orange', 1.0), facecolor='none')
        ax.add_patch(rect)
        for index in data["subsample/rand"]:
            j = int((index-1)/32)
            i = (index-1)%32
            #rect = patches.Rectangle((i-0.50, j-0.50), 1, 1, linewidth=0.25, edgecolor=_alpha_to_solid_on_bg('blue', 1.0), facecolor='none')
            #ax.add_patch(rect)

            ax.scatter(i,j, s=10, c='blue', marker='x')

    for ax in axs[1,0:2]:
        ax.invert_yaxis()

    fig.colorbar(pos, ticks=[0,1,2,3], ax=axs[0,3])

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig('/{}/plots/avalanche_sandpile_illustration.pdf'.format(base_path))



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
