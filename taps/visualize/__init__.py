

class view:
    def __init__(self, paths=None, viewer='Plotter', savefig=None,
                 filename=None, gaussian=False, energy_paths=True, run=True,
                 **kwargs):
        self.paths = paths
        self.viewer = viewer
        self.savefig = savefig
        self.filename = filename
        self.gaussian = gaussian
        self.energy_paths = energy_paths
        self.kwargs = kwargs
        if run:
            self()

    def __call__(self, paths=None, viewer=None, savefig=None, filename=None,
                 gaussian=None, energy_paths=None, **kwargs):
        paths = paths or self.paths
        viewer = viewer or self.viewer
        savefig = savefig or self.savefig
        filename = filename or self.filename
        gaussian = gaussian or self.gaussian
        energy_paths = energy_paths or self.energy_paths
        kwargs = kwargs or self.kwargs

        from_ = 'taps.visualize.' + viewer.lower()
        module = __import__(from_, {}, None, [viewer])
        plotter = getattr(module, viewer)(**kwargs)
        plotter.plot(paths, savefig=savefig, filename=filename,
                     gaussian=gaussian, energy_paths=energy_paths)
        return plotter
