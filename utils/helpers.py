import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Timer(object):
    def __init__(self, print=True):
        self._print = print
        self._ticlist = []
        self._toclist = []
        self._elapsed = []

    def tic(self):
        self._ticlist.append(time.time())

    def toc(self):
        self._toclist.append(time.time())
        assert len(self._ticlist) == len(self._toclist)
        elapsed = self._toclist[-1] - self._ticlist[-1] 
        self._elapsed.append(elapsed)
        if self._print:
            print("Elapsed time: ", elapsed)
    
    @property
    def ticlist(self):
        return self._ticlist
    
    @property
    def toclist(self):
        return self._toclist
    
    @property
    def elapsed(self):
        return self._elapsed


## === Matplotlib helpers ===
def integer_histogram(data):
    """Plot discrete histogram.
    """
    series = pd.Series(data)
    value_counts = series.value_counts().sort_index()
    ax = value_counts.plot(kind='bar', rot=0)
    ax.set_yticks(np.arange(0, value_counts.max() + 1, 5)) 
    fig = ax.get_figure()
    return fig

def plot_colorbar(min_depth, max_depth, plot=True):
    """Plot colorbar for depth map.

    Args:
        min_depth (float)
        max_depth (float)
        plot (bool, optional): If True, plot the image. If False, return the image. 
            Defaults to True.
    """
    fig, ax = plt.subplots()
    # ax.imshow(img_match)
    img = plt.imshow(np.array([[min_depth, max_depth]]), cmap="jet")
    img.set_visible(False)
    img.axes.set_visible(False)
    plt.colorbar(orientation="vertical")
    if plot:
        plt.show()
    else:
        return fig

def make_manager(fig):
    """
    Create a dummy figure and use its manager to display "fig".
    Example:
        fig = safe_load_pickle("fig.pkl")
        make_manager(fig)
        fig.show()
    Reference:
        https://gist.github.com/demisjohn/883295cdba36acbb71e4?permalink_comment_id=3177271#gistcomment-3177271
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
