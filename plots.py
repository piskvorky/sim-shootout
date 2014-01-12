import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

FLANN1000 = [ # results for k=1000
    "flann@0.7 flann@0.9 flann@0.95 flann@0.99".split(),  # method accuracy settings
    [1.404, 1.569, 1.839, 1.596],  # timings
    [0.185, 0.330, 0.371, 0.367],  # precicions
    [0.268, 0.129, 0.120, 0.113],  # avg diffs
    [0.205, 0.127, 0.0, 0.119],  # stddev diff
    [0.947, 0.781, 0.818, 0.801],  # max diff
]
FLANN1000_BATCH = [0.244, 0.342, 0.396, 0.337]  # batch timings (=resolving all 100 queries at once)
FLANN1000_SHORT = [[row[0], row[-1]] for row in FLANN1000]

FLANN100 = [
    "flann@0.7 flann@0.9 flann@0.95 flann@0.99".split(),
    [0.350, 0.369, 0.409, 0.367],
    [0.133, 0.248, 0.254, 0.239],
    [0.257, 0.116, 0.119, 0.112],
    [0.212, 0.114, 0.0, 0.1],
    [0.906, 0.816, 0.885, 0.866],
]
FLANN100_BATCH = [0.056, 0.049, 0.055, 0.049]
FLANN100_SHORT = [[row[0], row[-1]] for row in FLANN100]

FLANN10 = [
    "flann@0.7 flann@0.9 flann@0.95 flann@0.99".split(),
    [0.258, 0.275, 0.297, 0.334],
    [0.234, 0.351, 0.381, 0.366],
    [0.092, 0.045, 0.044, 0.042],
    [0.112, 0.053, 0.0, 0.048],
    [0.835, 0.707, 0.473, 0.265],
]
FLANN10_BATCH = [0.041, 0.024, 0.027, 0.027]
FLANN10_SHORT = [[row[0], row[-1]] for row in FLANN10]

FLANN1 = [
    "flann@0.7 flann@0.9 flann@0.95 flann@0.99".split(),
    [0.213, 0.279, 0.321, 0.296],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
]
FLANN1_BATCH = [0.035, 0.031, 0.038, 0.033]
FLANN1_SHORT = [[row[0], row[-1]] for row in FLANN1]

ANNOY1000 = [
    "annoy@1 annoy@10 annoy@50 annoy@100 annoy@500".split(),
    [1.030, 7.439, 34.163, 64.832, 287.705],
    [0.048, 0.300, 0.703, 0.849, 0.992],
    [0.489, 0.112, 0.017, 0.006, 0.0],
    [0.193, 0.085, 0.020, 0.010, 0.001],
    [0.929, 0.482, 0.111, 0.101, 0.015],
]

ANNOY100 = [
    "annoy@1 annoy@10 annoy@50 annoy@100 annoy@500".split(),
    [0.403, 1.083, 4.186, 8.026, 36.598],
    [0.045, 0.168, 0.496, 0.664, 0.939],
    [0.404, 0.152, 0.036, 0.018, 0.002],
    [0.225, 0.126, 0.039, 0.024, 0.005],
    [0.870, 0.563, 0.206, 0.126, 0.075],
]

ANNOY10 = [
    "annoy@1 annoy@10 annoy@50 annoy@100 annoy@500".split(),
    [0.395, 0.481, 0.958, 1.552, 6.592],
    [0.160, 0.230, 0.391, 0.501, 0.824],
    [0.175, 0.117, 0.052, 0.032, 0.006],
    [0.163, 0.126, 0.064, 0.043, 0.013],
    [0.870, 0.554, 0.364, 0.192, 0.075],
]

ANNOY1 = [
    "annoy@1 annoy@10 annoy@50 annoy@100 annoy@500".split(),
    [0.396, 0.479, 0.649, 0.851, 2.728],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]

SKLEARN1 = 0.176
SKLEARN10 = 2142.822
SKLEARN100 = 2359.800

GENSIM = 678.671
GENSIM_BATCH = 353.908

max_avgdiff = max(max(res[3]) for res in [ANNOY1, ANNOY10, ANNOY100, ANNOY1000, FLANN1, FLANN10, FLANN100, FLANN1000])
min_prec = min(min(res[2]) for res in [ANNOY1, ANNOY10, ANNOY100, ANNOY1000, FLANN1, FLANN10, FLANN100, FLANN1000])

def annotate_points(ax, labels, xs, ys, colour, x_offset, marker='o', alternate=False):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.scatter(xs, ys, s=80, marker=marker, c=colour, alpha=0.5)
    s = sorted(range(len(xs)), key=lambda p: xs[p])
    up_downs = [s.index(pos) % 2 if alternate else 0 for pos in range(len(xs))]
    for num, (label, x, y) in enumerate(zip(labels, xs, ys)):
        ax.annotate(
            label,
            xy=(x, y), xytext=(0, [-6, 6][up_downs[num]]), fontsize=12, color=colour, alpha=0.8,
            textcoords='offset points', ha=x_offset, va=['top', 'bottom'][up_downs[num]],
            # bbox = dict(boxstyle='round,pad=0.5', fc=colour, alpha=0.1),
            # arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
        )


def plot_results(methods, loc='best', colours=('r', 'g', 'b'), alternate=False, log=False):
    plt.locator_params(axis='y', nbins=10)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    if len(methods) != 2:
        left_rights = ['center'] * len(methods)
    else:
        left_rights = [['left', 'right'][pos % 2] for pos in range(len(methods))]

    max_avgdiff, min_prec = 0.0, 1.0
    for pos, (labels, timings, precs, avgdiffs, stddevdiffs, maxdiff) in enumerate(methods):
        assert len(labels) == len(avgdiffs) == len(precs) == len(timings) == len(maxdiff) == len(stddevdiffs)
        annotate_points(ax1, labels, timings, avgdiffs, x_offset=left_rights[pos], colour=colours[pos], alternate=alternate) #ymax=max_avgdiff)
        annotate_points(ax2, labels, timings, precs, x_offset=left_rights[pos], colour=colours[pos], alternate=alternate) #ymin=min_prec
        max_avgdiff, min_prec = max(max(avgdiffs), max_avgdiff), min(min(precs), min_prec)
    # ax1.legend(loc=loc)

    # make the y axis for avgdiff go from 0.0 to max(avg diffs)
    extra = (max_avgdiff + 0.001) / (15 * 2)
    ax1.set_ylim(bottom=-2 * extra, top=max_avgdiff + extra)
    # ax1.yaxis.set_ticks(np.arange(0.0, max_avgdiff + 0.001, (max_avgdiff + 0.001) / 10.0))  # 10 ticks on the y axis
    ax1.yaxis.set_major_locator(plt.MaxNLocator(15))

    # make the y axis for precision go from min(precision) to 1.0
    extra = (1 - min_prec + 0.001) / (15 * 2)
    ax2.set_ylim(bottom=min_prec - 2 * extra, top=1.0 + extra)
    # ax2.yaxis.set_ticks(np.arange(1.0, min_prec - 0.001, -(1.0 - min_prec + 0.001) / 10.0))  # 10 ticks on the y axis
    ax2.yaxis.set_major_locator(plt.MaxNLocator(15))

    fig.tight_layout() #pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.subplots_adjust(wspace=.2)

    if log:
        ax1.set_xscale('log')
    ax1.set_xlabel("ms/query")
    ax1.set_ylabel("avg diff")

    if log:
        ax2.set_xscale('log')
    ax2.set_xlabel("ms/query")
    ax2.set_ylabel("avg precision")

    # fig.suptitle(title)
    return fig

def generate_figures():
    plot_results([FLANN1], alternate=True, loc="center right").savefig("flann1.png", bbox_inches='tight')
    plot_results([FLANN10], alternate=True).savefig("flann10.png", bbox_inches='tight')
    plot_results([FLANN100], alternate=True).savefig("flann100.png", bbox_inches='tight')
    plot_results([FLANN1000], alternate=True).savefig("flann1000.png", bbox_inches='tight')

    plot_results([ANNOY1], alternate=True, loc="center right").savefig("annoy1.png", bbox_inches='tight')
    plot_results([ANNOY10], log=False).savefig("annoy10.png", bbox_inches='tight')
    plot_results([ANNOY100]).savefig("annoy100.png", bbox_inches='tight')
    plot_results([ANNOY1000]).savefig("annoy1000.png", bbox_inches='tight')

    plot_results([ANNOY1, FLANN1_SHORT], alternate=True, loc="center right").savefig("flann_annoy1.png", bbox_inches='tight')
    plot_results([ANNOY10, FLANN10_SHORT], log=False).savefig("flann_annoy10.png", bbox_inches='tight')
    plot_results([ANNOY100, FLANN100_SHORT], log=False).savefig("flann_annoy100.png", bbox_inches='tight')
    plot_results([ANNOY1000, FLANN1000_SHORT], log=False).savefig("flann_annoy1000.png", bbox_inches='tight')

def generate_table():
    for method in [ANNOY1, ANNOY10, ANNOY100, ANNOY1000, FLANN1, FLANN10, FLANN100, FLANN1000]:
        for label, timing, precision, avgdiff, stddevdiff, maxdiff in zip(*method):
            print '<tr><td>%s</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.3f</td><td>%.2f</td><td>%.1f</td><td>%s</td><td>%s</td></tr>' %\
                (label, '', precision, avgdiff, stddevdiff, maxdiff, timing, '', '')
            print
