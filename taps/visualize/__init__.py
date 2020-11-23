from taps.visualize.plotter import Plotter


def view(paths, viewer='plotter', savefig=None, filename=None, gaussian=False,
         energy_paths=True, **kwargs):

    return Plotter(**kwargs).plot(paths, savefig=savefig, filename=filename,
                                  gaussian=gaussian, energy_paths=energy_paths)
