from taps.visualize.plotter import Plotter


def view(paths, viewer='plotter', **kwargs):

    return Plotter(**kwargs).plot(paths)
