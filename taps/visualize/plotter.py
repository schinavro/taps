import os
import pickle
import colorsys
import matplotlib
import numpy as np
from ase.data.colors import jmol_colors
from matplotlib import pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
from taps.utils.shortcut import isStr, isstr, istpl, isBool, isInt, isint, isSclr
from taps.utils.shortcut import isflt, islst, isArr, isTpl, issclr
from taps.utils.shortcut import isarr, asst, dflt


# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rc('font', family='sans-serif')


def setfont(font):
    return r'\font\a %s at 14pt\a ' % font


def lighten_color(color, amount=0.2):
    """
    Lightens the given color by multiplying (1-luminosity) by the given
    amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames.get(color, color)
    except TypeError:
        c = mc.cnames.get(tuple(color), tuple(color))
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class Plotter:
    plotter_parameters = {
        'mapfile': {dflt: "'plotter_map.pkl'", asst: isStr},
        'calculate_map': {dflt: "False", asst: isBool},
        'save_format': {dflt: "'svg'", asst: isStr},
        'savefig': {'default': "False", 'assert': isBool},
        'filename': {'default': "'plotter'", 'assert': isStr},
        'translation': {'default': "0.", 'assert': 'True'},
        'conformation': {'default': "1.", 'assert': '{name:s} > 0'},
        'pbc': {'default': "np.array([True, True, True])", 'assert': 'True'},
        'line_color': {'default': "'r'", 'assert': isstr},
        'energy_range': {'default': "(-100, 100)", 'assert': isTpl},
        'energy_digit': {'default': "-2", 'assert': isint},
        'quiver_scale': {'default': "None", 'assert': issclr},

        'lgd_ftsz': {'default': "None", 'assert': isint},
        'tick_size': {'default': "None", 'assert': isint},
        'plot_along_distance': {'default': 'True', 'assert': isBool},
        'prj': {'default': "None", 'assert': 'True'},

        'ttl2d': {'default': "'Potential Energy Surface'", 'assert': isstr},
        'fgsz2d': {'default': "(7.5, 6)", 'assert': istpl},
        'xlbl2d': {'default': "r'$x$'", 'assert': isstr},
        'ylbl2d': {'default': "r'$y$'", 'assert': isstr},
        'xlbl2dftsz': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ylbl2dftsz': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'xlim2d': {'default': "None", 'assert': isarr},
        'ylim2d': {'default': "None", 'assert': isarr},
        'rngX2d': {'default': "36", 'assert': isInt},
        'rngY2d': {'default': "36", 'assert': isInt},
        'alp2d': {'default': "None", 'assert': isflt},
        'cmp2d': {'default': "None", 'assert': isstr},
        'lvls2d': {'default': "None", 'assert': islst},
        'inln2d': {'default': "True", 'assert': isBool},
        'ftsz2d': {'default': "None", 'assert': isint},
        'ftsz2dInfo': {'default': "None", 'assert': isint},
        'fmt2d': {'default': "'%.2f'", 'assert': isstr},
        'pthsClr2d': {'default': "None", 'assert': 'True'},

        'ttlGMu': {dflt: "'Gaussian Potential Energy Surface'", asst: isstr},
        'fgszGMu': {'default': "(7.5, 6)", 'assert': istpl},
        'xlblGMu': {'default': "r'$x$'", 'assert': isstr},
        'ylblGMu': {'default': "r'$y$'", 'assert': isstr},
        'ftszGMuXlbl': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ftszGMuYlbl': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ftszGMuMx': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'xlimGMu': {'default': "None", 'assert': isarr},
        'ylimGMu': {'default': "None", 'assert': isarr},
        'cmpGMu': {'default': "None", 'assert': isstr},
        'lvlsGMu': {'default': "None", 'assert': islst},
        'inlnGMu': {'default': "True", 'assert': isBool},
        'ftszGMu': {'default': "None", 'assert': isint},
        'fmtGMu': {'default': "'%.2f'", 'assert': isstr},
        'ftszGMuClrbr': {'default': "None", 'assert': isflt},

        'ttlGCov': {'default': "'Uncertainty Map'", 'assert': isstr},
        'fgszGCov': {'default': "(7.5, 6)", 'assert': istpl},
        'xlblGCov': {'default': "r'$x$'", 'assert': isstr},
        'ylblGCov': {'default': "r'$y$'", 'assert': isstr},
        'xlimGCov': {'default': "None", 'assert': isarr},
        'ylimGCov': {'default': "None", 'assert': isarr},
        'ftszGCovXlbl': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ftszGCovYlbl': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ftszGCovMx': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'cmpGCov': {'default': "None", 'assert': isstr},
        'lvlsGCov': {'default': "None", 'assert': islst},
        'ftszGCovClrbr': {'default': "None", 'assert': isflt},

        'ttlE': {'default': "'Energy'", 'assert': isstr},
        'fgszE': {'default': "(6, 3)", 'assert': istpl},
        'ftszE': {'default': "None", 'assert': isflt},
        'xlblE': {'default': "'Path distance'", 'assert': isstr},
        'ylblV': {'default': "r'Potential Energy'", 'assert': isstr},
        'ylblT': {'default': "r'Kinetic Energy'", 'assert': isstr},
        'ftszEXlbl': {'default': "13", 'assert': isint},
        'ftszVYlbl': {'default': "13", 'assert': isint},
        'ftszTYlbl': {'default': "13", 'assert': isint},
        'mrkrV': {'default': "'-'", 'assert': isstr},
        'mrkrH': {'default': "'--'", 'assert': isstr},
        'mrkrT': {'default': "'r:'", 'assert': isstr},
        'ylimHE': {'default': "None", 'assert': istpl},
        'ylimTE': {'default': "None", 'assert': istpl},
        'alpGV': {'default': "0.2", 'assert': isflt},
        'alpGH': {'default': "0.2", 'assert': isflt},
        'fgszLgnd': {'default': "None", 'assert': istpl},
        'ftszLgnd': {'default': "None", 'assert': isflt},

        'ttl3d': {'default': "'Potential Energy Surface'", 'assert': isstr},
        'fgsz3d': {'default': "None", 'assert': 'True'},
        'xlbl3d': {'default': "r'$x$'", 'assert': isstr},
        'ylbl3d': {'default': "r'$y$'", 'assert': isstr},
        'zlbl3d': {'default': "r'$z$'", 'assert': isstr},
        'xlbl3dftsz': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'ylbl3dftsz': {dflt: "13", 'assert': isint + ' or ' + isStr},
        'zlbl3dftsz': {dflt: "13", 'assert': isint + ' or ' + isStr},

        'xlim3d': {'default': "np.array([0, 1])", 'assert': "True"},
        'ylim3d': {'default': "np.array([0, 1])", 'assert': 'True'},
        'zlim3d': {'default': "np.array([0, 1])", 'assert': 'True'},
    }

    def __init__(self, filename=None, prj=None, prjf=None,
                 **kwargs):
        self.filename = filename
        self.prj = prj
        self.prjf = prjf
        for key in self.plotter_parameters.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif self.__dict__.get(key) is not None:
                continue
            else:
                setattr(self, key, None)

    def __setattr__(self, key, value):
        if key == 'prj':
            if value is None:
                def value(x):
                    return x
            super().__setattr__(key, value)
        elif key == 'prjf':
            if value is None:
                def value(f, x):
                    return f
            super().__setattr__(key, value)
        elif key in self.plotter_parameters:
            default = self.plotter_parameters[key]['default']
            assertion = self.plotter_parameters[key]['assert']
            if value is None:
                value = eval(default.format(name='value'))
            assert eval(assertion.format(name='value')), (key, value)
            super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        else:
            raise AttributeError('%s not implemented for %s' % (key, value))

    def plot(self, paths, savefig=None, filename=None, gaussian=False,
             energy_paths=True):
        if filename is None:
            filename = self.filename
        if savefig is None:
            savefig = self.savefig
        dir = os.path.dirname(filename)
        if dir == '':
            dir = '.'
        if not os.path.exists(dir):
            os.makedirs(dir)
        p = paths.coords
        if self.xlim2d is None:
            self.xlim2d = np.array([p[0].min(), p[0].max()])
        if self.ylim2d is None:
            self.ylim2d = np.array([p[1].min(), p[1].max()])
        D, M = self.get_shape(self.prj(paths.coords))
        if D == 1:
            raise NotImplementedError('No 1D plot')
        elif D == 2:
            self.plot_2D(paths, savefig, filename)
        elif D == 3:
            self.plot_3D(paths, savefig, filename)
        else:
            raise NotImplementedError("Can't plot ")
        if gaussian:
            self.plot_gaussian(paths, savefig, filename, gaussian)
        if energy_paths:
            self.plot_energy_paths(paths, savefig, filename, gaussian)

    def plot_3D(self, paths, savefig, filename):
        Axes3D  # silence!
        fig = plt.figure(figsize=self.fgsz3d)
        ax = fig.add_subplot(111, projection='3d')
        self._fig, self._ax = fig, ax
        ax.set_title(self.ttl3d)
        ax.set_xlabel(self.xlbl3d, fontsize=self.xlbl3dftsz)
        ax.set_ylabel(self.ylbl3d, fontsize=self.ylbl3dftsz)
        ax.set_zlabel(self.zlbl3d, fontsize=self.zlbl3dftsz)

        ax.set_xlim(self.display_lim(self.xlim3d))
        ax.set_ylim(self.display_lim(self.ylim3d))
        ax.set_zlim(self.display_lim(self.zlim3d))

        self.plot_trajectory(paths, plt, ax)

        if savefig:
            plt.tight_layout()
            with open(filename + '_fig3D.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_3D.' + self.save_format,
                        format=self.save_format)

    def plot_2D(self, paths, savefig, filename):
        fig, ax = plt.subplots(figsize=self.fgsz2d)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
        # ax.set_title(self.ttl2d)
        # ax.set_xlabel(self.xlbl2d, fontsize=self.xlbl2dftsz)
        # ax.set_ylabel(self.ylbl2d, fontsize=self.ylbl2dftsz)

        plt.xlim(self.display_lim(self.xlim2d))
        plt.ylim(self.display_lim(self.ylim2d))

        X, Y = self.get_meshgrid(grid_type='contour')

        if self.calculate_map:
            Z = self.calculate_model_map(paths, model_type='real')
            Z.reshape((self.rngX2d, self.rngY2d))
            with open(self.mapfile, 'wb') as f:
                pickle.dump(Z, f)
        elif os.path.exists(self.mapfile):
            with open(self.mapfile, 'rb') as f:
                Z = pickle.load(f)
            if Z.shape != X.shape:
                range_x, range_y = Z.shape
                X, Y = self.get_meshgrid(grid_type='contour',
                                         range_x=range_x, range_y=range_y)
        else:
            Z = np.zeros((self.rngX2d, self.rngY2d))
        ctrkwargs = {}
        if self.lvls2d is not None:
            ctrkwargs['levels'] = self.lvls2d
        CS = ax.contourf(X, Y, self.display_map(Z, map_type='contour'),
                         cmap=self.cmp2d, corner_mask=True,
                         **ctrkwargs)

        fig.colorbar(CS)

        self.plot_trajectory(paths, plt, ax)
        self.plot_information(paths, plt, ax, information='finder')

        if savefig:
            plt.tight_layout()
            with open(filename + '_fig2D.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_2D.' + self.save_format,
                        format=self.save_format)

    def plot_gaussian(self, paths, savefig, filename, gaussian):
        if not gaussian:
            return
        fig, ax = plt.subplots(figsize=self.fgszGMu)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)

        # ax.set_title(self.ttlGMu)

        # ax.set_xlabel(self.xlblGMu, fontsize=self.ftszGMuXlbl)
        # ax.set_ylabel(self.ylblGMu, fontsize=self.ftszGMuYlbl)

        d_xlim = self.display_lim(self.xlim2d)
        d_ylim = self.display_lim(self.ylim2d)
        ax.set_xlim(d_xlim)
        ax.set_ylim(d_ylim)

        ave_map = self.calculate_model_map(paths, model_type='ave_map')
        cov_map = self.calculate_model_map(paths, model_type='cov_map')
        self._ave_map = ave_map
        self._cov_map = cov_map

        X, Y = self.get_meshgrid(grid_type='contour')
        # cax = fig.add_axes()

        CS = ax.contourf(X, Y, self.display_map(ave_map, map_type='contour'),
                         cmap=self.cmpGMu, levels=self.lvlsGMu)
        # ax.clabel(CS, inline=self.inlnGMu, fontsize=self.ftszGMu,
        #           fmt=self.fmtGMu)
        # im = ax.pcolormesh(X, Y, self.display_map(ave_map), cmap=self.cma2,
        #                    vmin=-150, vmax=100)
        # fig.colorbar(im, cax=cax)
        cbar = fig.colorbar(CS)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=self.ftszGMuClrbr)
        # cbar.ax.tick_params(labelsize=self.ftszGMuClrbr)

        #  forces = paths.model.get_forces(paths, index=np.s_[:])
        self.plot_trajectory(paths, plt, ax,
                             xlim=self.xlimGMu, ylim=self.ylimGMu)
        self.plot_data(paths, plt, ax)
        self.plot_information(paths, plt, ax, information='finder')
        self.plot_info_map(paths, plt, ax, information='maximum_energy')

        if savefig:
            # plt.tight_layout()
            with open(filename + '_figmu.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_mu.' + self.save_format,
                        format=self.save_format)

        fig, ax = plt.subplots(figsize=self.fgszGCov)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)

        # ax.set_title(self.ttlGCov)
        ax.set_xlim(d_xlim)
        ax.set_ylim(d_ylim)
        # ax.set_xlabel(self.xlblGCov, fontsize=self.ftszGCovXlbl)
        # ax.set_ylabel(self.ylblGCov, fontsize=self.ftszGCovYlbl)

        im = ax.contourf(X, Y, self.display_map(cov_map, map_type='contour'),
                         cmap=self.cmpGCov, levels=self.lvlsGCov)
        # im = ax.pcolormesh(X, Y, self.display_map(cov_map), vmax=2, vmin=0,
        #                    cmap=self.cma3)
        # fig.colorbar(im, cax=cax)
        cbar2 = fig.colorbar(im)
        ticklabs2 = cbar2.ax.get_yticklabels()
        cbar2.ax.set_yticklabels(ticklabs2, fontsize=self.ftszGCovClrbr)
        # cbar2.ax.tick_params(labelsize=self.ftszGCovClrbr)
        self.plot_trajectory(paths, plt, ax,
                             xlim=self.xlimGCov, ylim=self.ylimGCov)
        self.plot_data(paths, plt, ax, mark_update=True)
        self.plot_information(paths, plt, ax, information='gaussian')
        self.plot_info_map(paths, plt, ax, information='maximum_uncertainty')
        if savefig:
            # plt.tight_layout()
            with open(filename + '_figcov.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_cov.' + self.save_format,
                        format=self.save_format)

    def plot_energy_paths(self, paths, savefig, filename, gaussian):
        Vunit, Kunit = '', ''
        if paths.real_model.potential_unit != 'unitless':
            Vunit = '$(%s)$' % paths.model.potential_unit
        Kunit = '$(%s)$' % (paths.coords.unit or 'unitless')

        fig, ax = plt.subplots(figsize=self.fgszE)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
        if self.ylimHE is not None:
            ax.set_ylim(self.ylimHE)
        ttlE = ''
        formatkwargs = {'x': 'dist', 'pf': 'paths.finder'}
        for key, value in paths.finder.display_graph_title_parameters.items():
            formatkwargs['key'] = key
            if not eval(value['under_the_condition'].format(**formatkwargs)):
                continue
            label = value['label']
            unit = eval(value.get('unit', '""').format(p='paths'))
            if unit == 'unitless' or unit is None:
                unit = ''
            f = eval(value['value'].format(**formatkwargs))
            number = self.display_float(f, unit=unit)
            ttlE += label + r': ${n:s}$'.format(n=number) + ';   '
        if ttlE == '':
            ttlE = self.ttlE
        else:
            ttlE = ttlE[:-3]
        # ax.set_title(ttlE)
        # ax.set_xlabel(self.xlblE, fontsize=self.ftszEXlbl)
        # ax.set_ylabel(self.ylblV + Vunit, fontsize=self.ftszVYlbl)
        ax.set_ylabel('Total & Potential', fontsize=self.ftszVYlbl)

        dist = paths.get_displacements(index=np.s_[1:-1])
        V = paths.get_potential_energy(index=np.s_[1:-1])
        T = paths.get_kinetic_energy(index=np.s_[1:-1])
        lns = ax.plot(dist, V, self.mrkrV, label='$V$')
        if gaussian:
            cov_coords = paths.get_covariance(index=np.s_[1:-1])
            cov_coords[cov_coords < 0] = 0
            color = lighten_color(lns[0].get_color())
            ax.fill_between(dist, V + cov_coords, V - cov_coords,
                            color=color, label=r'$\Sigma$')
            cov_max_idx = cov_coords.argmax()
            annot_x = dist[cov_max_idx]
            annot_y = V[cov_max_idx] + cov_coords.max()
            # ax.annotate(s=r'$\Sigma_{max}$', xy=(annot_x, annot_y),
            #             xytext=(0, 16), textcoords='offset points',
            #             ha='center', va='center',
            #             fontsize=13, arrowprops=dict(arrowstyle='->'))
        H = V + T
        # ax.plot(dist, H, '--')
        ax2 = ax.twinx()
        if self.ylimTE is not None:
            ax2.set_ylim(self.ylimTE)
        ax2.tick_params(axis='both', which='major', labelsize=self.tick_size)
        # ax2.set_ylabel(self.ylblT + Kunit, fontsize=self.ftszTYlbl)
        ax2.set_ylabel(self.ylblT, fontsize=self.ftszTYlbl)
        if self.plot_along_distance:
            lns += ax.plot(dist, H, self.mrkrH, label='$H$')
            lns += ax2.plot(dist, T, self.mrkrT, label='$T$')
        else:
            lns += ax.plot(H, self.mrkrH, label='$H$')
            lns += ax2.plot(T, self.mrkrT, label='$T$')
        # if gaussian:
        #     color = lighten_color(lns[-1].get_color())
        #     ax.fill_between(dist, H + uncertainty_coords, H - uncertainty_coords,
        #                     color=color)


        # Max E indicator
        x0 = dist[V.argmax()]
        y0 = V.min()
        x1 = x0
        y1 = V.max()
        dE = (V.max() - y0)
        dE = r'$%s$' % self.display_float(dE, unit=Vunit)
        # ax.annotate(s=r'$V_{max}$', xy=(x0, y1),
        #             xytext=(0, -16), textcoords='offset points',
        #             ha='center', va='center',
        #             fontsize=13, arrowprops=dict(arrowstyle='->'))
        for key, value in paths.finder.display_graph_parameters.items():
            formatkwargs['key'] = key
            plot = getattr(ax, value.get('plot', 'plot'))
            if not eval(value['under_the_condition'].format(**formatkwargs)):
                continue
            args = eval(value['args'].format(**formatkwargs))
            kwargs = eval(value['kwargs'].format(**formatkwargs))
            pltobj = plot(*args, **kwargs)
            if isinstance(pltobj, list):
                pltobj = pltobj[0]
            amnt = value.get('lighten_color_amount')
            if amnt is not None:
                pltobj.set_color(lighten_color(pltobj.get_color(), amount=amnt))
            if value.get('isLabeld', True):
                lns.append(pltobj)
        # sp1 = ax.transData.transform_point((x0, y0))
        # sp2 = ax.transData.transform_point((x1, y1))
        # rise = (sp2[1] - sp1[1])
        # run = (sp2[0] - sp1[0])
        # slope_degrees = np.degrees(np.arctan2(rise, run))
        # label=r'$\Sigma_{max}$',
        # ax.text(x1, (y0 + y1) / 2, dE, rotation=slope_degrees, size=13,
        #         horizontalalignment='center', verticalalignment='top')
        # ax.annotate(s=dE, xy=(x1, (y0 + y1) / 2), xytext=(10, 0), size=13,
        #             textcoords='offset points', rotation=slope_degrees,
        #             arrowprops=dict(arrowstyle='<->'),
        #             horizontalalignment='left',
        #             verticalalignment='bottom', label=r'$V_{max}$')
        # lnVMx.set_rotation(slope_degrees)
        if savefig:
            plt.tight_layout()
            with open(filename + '_figE.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_E.' + self.save_format,
                        format=self.save_format)

        # handles, labels = ax2.get_legend_handles_labels()
        # ax2.get_legend_handles_labels()
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper left', fontsize=self.ftszLgnd)

        if savefig:
            plt.tight_layout()
            with open(filename + '_figLgnd.pkl', 'wb') as f:
                pickle.dump(fig, f)
            plt.savefig(filename + '_Lgnd.' + self.save_format,
                        format=self.save_format)
            plt.close('all')

    def plot_trajectory(self, paths, plt, ax, xlim=None, ylim=None, zlim=None,
                        forces=None):
        plotter_coords = self.prj(paths.coords)
        D, M = self.get_shape(plotter_coords)
        if D * M < 4:
            M, D = 1, M * D
            coords = plotter_coords.reshape(D, 1, -1)
            line_color = [self.line_color]
            scatter_color = [lighten_color(self.line_color, amount=0.5)]
            # scatter_color = ['orange']
        else:
            coords = plotter_coords
            line_color = jmol_colors[paths._numbers]
            scatter_color = [lighten_color(c) for c in line_color]
        for i in range(M):
            coord = coords[:, i, :]
            d_coord = self.display_coord(coord, xlim=xlim, ylim=ylim, zlim=zlim)
            d_traj = self.periodic_masked_array(d_coord, xlim, ylim, zlim)
            # ax.plot(*d_path, color=color[i])
            # ax.scatter(*d_path, color='orange')
            ax.plot(*d_traj, color=line_color[i])
            ax.scatter(*d_coord, color=scatter_color[i], alpha=0.5)
            if forces is not None:
                force = forces[:, i, :]
                # *(forces.reshape(-1, paths.N))
                ax.quiver(*d_coord, *force, color='w')

    def plot_data(self, paths, plt, ax, mark_update=False, quiver_scale=None):
        if quiver_scale is None and self.quiver_scale is None:
            quiver_scale = 1 / self.conformation
        elif quiver_scale is None:
            quiver_scale = self.quiver_scale

        D, N = paths.coords.shape
        data = paths.model.get_data(paths)
        X_dat = self.display_coord(data['X'].reshape(D, -1))
        F_dat = data['F'].reshape(D, -1)
        tX, tY = X_dat[0, :], X_dat[1, :]
        ax.scatter(tX, tY, color='black', marker='x', s=paths.coords.N)
        ax.quiver(tX, tY, *F_dat, color='w', angles='xy', scale_units='xy',
                  scale=quiver_scale)
        if mark_update:
            ax.scatter(tX[-1], tY[-1], color='red', marker='X', s=paths.N)

    def plot_information(self, paths, plt, ax, information='finder', xlim=None,
                         ylim=None):
        if xlim is None:
            xlim = self.display_lim(self.xlim2d)
        if ylim is None:
            ylim = self.display_lim(self.ylim2d)
        if information == 'finder':
            param = r''
            for key, value in paths.finder.display_map_parameters.items():
                number = eval(value['value'].format(pf='paths.finder'))
                unit = eval(value.get('unit', '""').format(p='paths'))
                if unit == 'unitless' or unit is None:
                    unit = ''
                force_LaTex = value.get('force_LaTex', False)
                significant_digit = value.get('significant_digit')
                df_kwargs = {'unit': unit, 'force_latex': force_LaTex,
                             'significant_digit': significant_digit}
                number = self.display_float(number, **df_kwargs)
                param += value['label'] + r'$: {n:s}$'.format(n=number)
                param += '\n'
        elif information == 'gaussian':
            param = r''
            # display_digit = min(np.log10(np.abs(values)))
            for key, value in paths.model.hyperparameters.items():
                if key == 'sigma_f':
                    continue
                number = self.display_float(value, force_latex=True,
                                            significant_digit=2)
                param += r'$\{key:s}: {n:s}$'.format(key=key, n=number)
                param += '\n'
        if param != r'':
            param = param[:-1]

        ec = (0.5, 0.5, 0.5)
        fc = (1., 1., 1.)
        ax2 = plt.text(0.5, 2.1, param, size=self.ftsz2dInfo,
                       ha="left", va="top",
                       bbox=dict(boxstyle="round", ec=ec, fc=fc, alpha=0.5))
        plt.gcf().canvas.draw()
        box = ax2.get_window_extent().transformed(plt.gca().transData.inverted())
        x1 = xlim[-1] - (xlim[-1] - xlim[0]) / 40
        y0 = ylim[-1] - (ylim[-1] - ylim[0]) / 20
        x0 = x1 - box.width
        ax2.set_position([x0, y0, box.width, box.height])

    def plot_info_map(self, paths, plt, ax, information='maximum_energy'):
        if information == 'maximum_energy':
            E = paths.get_potential_energy(index=np.s_[1:-1])
            unit = paths.model.potential_unit
            string = r'$\mu^{(max)} : %s$' % self.display_float(E.max(),
                                                                unit=unit)

            xy = paths.coords[..., 1 + E.argmax()].flatten() * self.conformation
            fontsize = self.ftszGMuMx
        elif information == 'maximum_uncertainty':
            cov = paths.get_covariance()
            cov[cov < 0] = 0
            string = r'$\Sigma^{(max)} : %s$' % self.display_float(
                                                cov.max(), force_latex=True,
                                                significant_digit=2)
            xy = paths.coords[..., cov.argmax()].flatten() * self.conformation
            fontsize = self.ftszGCovMx

        offset = 64
        ec = (0.5, 0.5, 0.5)
        fc = (1., 1., 1.)

        ax.annotate(s=string, xy=xy, xytext=(-1.5 * offset, -offset),
                    textcoords='offset points', fontsize=fontsize,
                    bbox=dict(boxstyle="round", ec=ec, fc=fc, alpha=0.5),
                    arrowprops=dict(connectionstyle="arc3,rad=.3",
                                    arrowstyle='->'))

    def plot_legend(self, paths, plt, ax):
        ax3 = plt.subplot(111)
        box = ax.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        legend_x = 1
        legend_y = 0.5
        plt.legend([self.mrkrV, self.mrkrH, self.mrkrT], loc='center left',
                   bbox_to_anchor=(legend_x, legend_y), fontsize=self.lgd_ftsz)

    def periodic_masked_array(self, coord, xlim=None, ylim=None, zlim=None):
        """
        coord is D x P array
        """
        D, P = coord.shape
        conformation = self.conformation
        diff = self.display_size(xlim, ylim, zlim, dimension=D)
        diff *= 0.5 * conformation
        coord_diff = np.diff(coord)  # D x P - 1
        mask_coord = np.hstack([np.abs(coord_diff) > diff, np.zeros((D, 1))])
        mask = np.any(mask_coord, axis=0)
        idx = np.argwhere(mask).flatten()
        mask_pad = np.zeros((D, len(idx) * 3), dtype=bool)
        mask_pad[:, 1::3] = True
        periodic_mask = np.insert(np.zeros((D, P), dtype=bool),
                                  np.repeat(idx + 1, 3), mask_pad, axis=1)
        coord_ = coord[:, idx + 1]    # D x P'
        _coord = coord[:, idx]        # D x P'
        m_coord = mask_coord[:, idx]
        coord_sign = np.sign(coord_diff[:, idx])
        coord_pad = np.zeros((D, len(idx) * 3))

        patch = m_coord * coord_sign * diff * 2 * conformation
        coord_pad[:, ::3] = coord_ - patch
        coord_pad[:, 2::3] = _coord + patch
        periodic_coord = np.insert(coord, np.repeat(idx + 1, 3), coord_pad,
                                   axis=1)
        return np.ma.MaskedArray(periodic_coord, periodic_mask)

    def display_window(self, xlim=None, ylim=None, zlim=None, dimension=3):
        if xlim is None:
            if dimension == 2:
                xlim = self.xlim2d
            elif dimension == 3:
                xlim = self.xlim3d
        if ylim is None:
            if dimension == 2:
                ylim = self.ylim2d
            elif dimension == 3:
                ylim = self.ylim3d
        if dimension == 3 and zlim is None:
            zlim = self.zlim3d
        return {0: xlim, 1: ylim, 2: zlim}

    def display_origin(self, xlim=None, ylim=None, zlim=None,
                       translation=None, dimension=3):
        if translation is None:
            translation = self.translation
        origin = np.zeros(dimension)
        win = self.display_window(xlim=xlim, ylim=ylim, zlim=zlim,
                                  dimension=dimension)
        if np.isscalar(translation):
            translation = np.array([translation] * dimension)
        for i in range(dimension):
            origin[i] = win[i][0] + translation[i]
        return origin

    def display_size(self, xlim=None, ylim=None, zlim=None, dimension=3):
        win = self.display_window(xlim, ylim, zlim, dimension)
        window_size = np.zeros((dimension, 1))
        for i in range(dimension):
            window_size[i] = (win[i][1] - win[i][0])
        return window_size

    def display_map(self, Z, map_type='pcolormesh'):
        """
        X : 2 x P
        Y : 2 x P
        Z : 2 x P
        """
        if map_type == 'pcolormesh':
            return Z
        elif map_type == 'contour':
            return Z
            # _Z = np.vstack([Z, Z[0]])
            # _Z = np.hstack([_Z, _Z[:, -1, np.newaxis]])
            # return _Z
        else:
            NotImplementedError('only `contour` `pcolormesh` support')

    def display_lim(self, lim, translation=None, conformation=None):
        if lim is None:
            return None
        elif type(lim) == list:
            lim = np.array(lim)
        if translation is None:
            translation = self.translation
        if conformation is None:
            conformation = self.conformation
        return conformation * (lim + translation)

    def display_coord(self, coord, xlim=None, ylim=None, zlim=None,
                      conformation=None, pbc=None):
        """
        coord : D x P
        """
        if conformation is None:
            conformation = self.conformation
        if pbc is None:
            pbc = self.pbc
        p = np.zeros(coord.shape)
        D = p.shape[0]
        origin = self.display_origin(xlim, ylim, zlim, dimension=D)
        win_size = self.display_size(xlim, ylim, zlim, dimension=D)
        for d in range(D):
            if pbc[d]:
                p[d] = (coord[d] - origin[d]) % (win_size[d]) + origin[d]
            else:
                p[d] = coord[d]
        return conformation * p

    def get_shape(self, coords):
        if len(coords.shape) == 2:
            return coords.shape[0], 1
        elif coords.shape == 3:
            return coords.shape[:2]
        else:
            shape = ','.join([str(d) for d in coords.shape])
            raise NotImplementedError('invalid shape (' + shape + ')')

    def get_meshgrid(self, grid_type='coords', xlim=None, ylim=None,
                     range_x=None, range_y=None):
        if xlim is None:
            xlim = self.xlim2d
        if ylim is None:
            ylim = self.ylim2d
        if range_x is None:
            range_x = self.rngX2d
        if range_y is None:
            range_y = self.rngY2d
        xlim = np.array(xlim) + self.translation
        ylim = np.array(ylim) + self.translation
        if grid_type == 'coords':
            x = np.linspace(*xlim, range_x)
            y = np.linspace(*ylim, range_y)
            X, Y = np.meshgrid(x, y)
            # return np.c_[X.ravel(), Y.ravel()].T[:, np.newaxis, :]
            return np.c_[X.ravel(), Y.ravel()].T
        elif grid_type in ['contour']:
            x = np.linspace(*xlim, range_x)
            y = np.linspace(*ylim, range_y)
            return self.conformation * np.array(np.meshgrid(x, y))
        elif grid_type in ['pcolormesh']:
            x = np.linspace(*xlim, range_x + 1)
            y = np.linspace(*ylim, range_y + 1)
            return self.conformation * np.array(np.meshgrid(x, y))
        else:
            NotImplementedError('Only `contour`, `pcolormesh` supports')

    def calculate_model_map(self, paths, model_type='real', xlim=None,
                            ylim=None, range_x=None, range_y=None,
                            grid_type='coords'):
        if range_x is None:
            range_x = self.rngX2d
        if range_y is None:
            range_y = self.rngY2d
        _coords = self.get_meshgrid(xlim=xlim, ylim=ylim, grid_type=grid_type)
        coords = self.prj(paths.coords(coords=_coords))
        shape = (range_x, range_y)
        if model_type == 'real':
            E = paths.get_potential_energy(coords=coords, real_model=True)
            return E.reshape(shape)
        elif model_type == 'ave_map':
            ave_map = paths.get_potential_energy(coords=coords)
            return ave_map.reshape(shape)
        elif model_type == 'cov_map':
            cov = paths.model.get_covariance(paths, coords=coords)
            cov_m = np.copy(np.diag(cov))
            cov_m[cov_m < 0] = 0
            cov_map = 1.96 * np.sqrt(cov_m) / 2
            return cov_map.reshape(shape)

    def display_float(self, f, display_range=None, display_digit=None,
                      leading_digit=None, last_digit=None,
                      significant_digit=None, unit='', force_latex=False):
        """
        get float
        return string
        """
        if display_range is None:
            display_range = self.energy_range
        if display_digit is None:
            display_digit = self.energy_digit
        if leading_digit is None:
            leading_digit = int(np.max(np.log10(np.abs(display_range))))
        lower, upper = display_range
        if last_digit is None:
            last_digit = np.min([display_digit, np.log10(upper - lower)])
            last_digit = int(last_digit)
        if significant_digit is None:
            significant_digit = leading_digit - last_digit

        if significant_digit > 6:
            # Overlap should be handled
            _f = np.sign(f) * (np.abs(f) % (10 ** (last_digit + 6)))
            if leading_digit > 6 or last_digit < -6 or force_latex:
                df = '{f:.6E}'.format(f=_f)
            elif last_digit >= 0:
                df = '{f:.f}'.format(f=_f)
            else:
                df = '{f:.{decimal:d}f}'.format(decimal=-last_digit, f=_f)
        elif significant_digit > 2:
            if leading_digit > 6 or last_digit < -6 or force_latex:
                df = '{f:.{sd:d}E}'.format(sd=significant_digit, f=f)
            elif last_digit >= 0:
                df = '{f:.f}'.format(f=f)
            else:
                df = '{f:.{decimal:d}f}'.format(decimal=-last_digit, f=f)
        else:
            if leading_digit > 4 or last_digit < -4 or force_latex:
                df = '{f:.2E}'.format(f=f)
            else:
                df = '{f:.2f}'.format(f=f)
        if "E" in df:
            base, exponent = df.split("E")
            df = r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        return df + unit


class FlatModelPlotter(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = True
        self.ttl2d = 'Flat Model Energy Surface'
        ref = -129.5
        self.xlim2d = np.array([-1.2, 1.2])
        self.ylim2d = np.array([-1.2, 1.2])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-1.2, 1.2])
        self.ylimGMu = np.array([-1.2, 1.2])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-0.2, 0.2, 15) + ref
        self.lvlsGMu = np.linspace(-0.2, 0.2, 15) + ref
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.pbc = np.array([False, False])
        self.energy_range = tuple(np.array([-0.5, 0.5]) + ref)
        self.quiver_scale = 3
        self.ylimHE = tuple(np.array([-0.5, 0.5]) + ref)
        self.ylimTE = (0, 1)

        super().__init__(**kwargs)


class AlanineDipeptidePlotter(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = False
        self.conformation = 180 / np.pi
        self.translation = 0.
        self.pbc = np.array([True, True])
        self.energy_digit = -2

        self.ttl2d = 'Alanine Dipeptide Potential Energy Surface'
        self.xlim2d = np.array([-np.pi, np.pi])
        self.ylim2d = np.array([-np.pi, np.pi])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-np.pi, np.pi])
        self.ylimGMu = np.array([-np.pi, np.pi])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-130, -129, 20)
        self.lvlsGMu = np.linspace(-130, -129, 20)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.energy_range = (-130, -129)
        self.ylimHE = (-130, -129)
        self.ylimTE = (0, 1)
        self.quiver_scale = 100

        super().__init__(**kwargs)


class MalonaldehydePlotter(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = False
        self.conformation = 1
        self.translation = 0.
        self.pbc = np.array([False, False])
        self.energy_digit = -2

        self.ttl2d = 'Malonaldehyde Potential Energy Surface'
        self.xlim2d = np.array([-0.6, 0.6])
        self.ylim2d = np.array([1.1, 1.4])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-0.6, 0.6])
        self.ylimGMu = np.array([1.1, 1.4])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-53.7, -52, 20)
        self.lvlsGMu = np.linspace(-53.7, -52, 20)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.energy_range = (-53.7, -52)
        self.ylimHE = (-53.7, -52)
        self.ylimTE = (0, 1)
        self.quiver_scale = 100

        super().__init__(**kwargs)


class HafniumDioxidePlotter(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = False
        self.conformation = 180 / np.pi
        self.translation = 0.
        self.pbc = np.array([True, True])
        self.energy_digit = -2

        self.ttl2d = 'Alanine Dipeptide Potential Energy Surface'
        self.xlim2d = np.array([-np.pi, np.pi])
        self.ylim2d = np.array([-np.pi, np.pi])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-np.pi, np.pi])
        self.ylimGMu = np.array([-np.pi, np.pi])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-130, -129, 20)
        self.lvlsGMu = np.linspace(-130, -129, 20)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.energy_range = (-130, -129)

        super().__init__(**kwargs)


class PeriodicModel2Plotter(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = True
        self.conformation = 180 / np.pi
        self.translation = 0.
        self.pbc = np.array([True, True])
        self.energy_digit = -2

        self.ttl2d = 'Alanine Dipeptide Potential Energy Surface'
        self.xlim2d = np.array([-np.pi, np.pi])
        self.ylim2d = np.array([-np.pi, np.pi])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-np.pi, np.pi])
        self.ylimGMu = np.array([-np.pi, np.pi])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-130, -129, 20)
        self.lvlsGMu = np.linspace(-130, -129, 20)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.energy_range = (-130, -129)
        self.quiver_scale = 100

        super().__init__(**kwargs)
