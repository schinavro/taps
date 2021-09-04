import numpy as np
from numpy import newaxis, concatenate, repeat, cross, cos, sin, sum
from numpy.linalg import norm
from taps.projectors import Projector


class AlaninedipeptideInverse(Projector):
    """
    Projector sends (phi, psi) -> 22 array coordinates

    Parameters
    ----------
    C70: array;
       Reference array of alaninedipeptide haing angle phi, psi (0, 0)
    """
    # 22 x 3
    C70 = np.array([[3.13042320, 8.69636925, 6.86034480],
                    [3.62171100, 7.75287090, 7.13119080],
                    [3.06158460, 6.94361235, 6.64450215],
                    [3.56832480, 7.62030150, 8.21956230],
                    [5.03124195, 7.76696535, 6.56486415],
                    [5.19044280, 7.93294500, 5.34189060],
                    [6.01497510, 7.58529855, 7.48734615],
                    [5.65167885, 7.45428060, 8.42656485],
                    [7.50000000, 7.50000000, 7.50000000],
                    [7.84286445, 8.35064955, 8.11409370],
                    [7.87916790, 6.20253030, 8.23173825],
                    [8.96770980, 6.15547755, 8.34399540],
                    [7.53870225, 5.32801485, 7.65999495],
                    [7.41872085, 6.16658700, 9.23056245],
                    [8.38912575, 7.62345720, 6.23286150],
                    [9.61384530, 7.55572530, 6.43116195],
                    [7.83695265, 7.81965300, 5.02677705],
                    [6.80236320, 7.87813665, 4.96104660],
                    [8.67108645, 7.97151630, 3.84856545],
                    [9.48211245, 8.68811745, 4.03978560],
                    [9.13567845, 7.01691960, 3.55615545],
                    [8.04737280, 8.33316525, 3.02266680]])

    def __init__(self, C70=None, **kwargs):
        self.C70 = C70 or self.C70

        super().__init__(**kwargs)

    @Projector.pipeline
    def x(self, coords):
        """
         coords :  2 x N
         returns : coords;  3 x A x N
        """
        phi, psi = coords
        p = self.C70.T[:, :, np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x A x N
        # Phi, Psi
        positions[:, :8] = self.rotate(p, phi, v=(6, 8), mask=np.s_[:8])
        positions[:, 14:] = self.rotate(p, psi, v=(14, 8), mask=np.s_[14:])
        return self.overlap_handler(positions)

    @Projector.pipeline
    def _x(self, coords):
        """
         coords :  2 x N
         returns : coords;  3 x A x N
        """
        phi, psi = coords
        p = self.C70.T[:, :, np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x A x N
        # Phi, Psi
        positions[:, :8] = self.rotate(p, phi, v=(6, 8), mask=np.s_[:8])
        positions[:, 14:] = self.rotate(p, psi, v=(14, 8), mask=np.s_[14:])
        return self.overlap_handler(positions)

    @Projector.pipeline
    def x_inv(self, coords):
        """
        Inverse function sends 3 x 22 x N -> 2 x N
         positions : 3 x A x N
         returns : coords;  3 x A x N, in this case 1 x 2 x N
        """

        phi = self.get_dihedral(coords, 4, 6, 8, 14)  # 1 x N
        psi = self.get_dihedral(coords, 6, 8, 14, 16)  # 1 x N
        return np.vstack([phi, psi])[np.newaxis, :]

    @Projector.pipeline
    def _x_inv(self, coords):
        """
        Inverse function sends 3 x 22 x N -> 2 x N
         positions : 3 x A x N
         returns : coords;  3 x A x N, in this case 1 x 2 x N
        """

        phi = self.get_dihedral(coords, 4, 6, 8, 14)  # 1 x N
        psi = self.get_dihedral(coords, 6, 8, 14, 16)  # 1 x N
        return np.vstack([phi, psi])[np.newaxis, :]

    @Projector.pipeline
    def f(self, forces, coords):
        """
        coords : D x A x N; 2 x 1 x P
        forces : D x A x P; 2 x 1 x P
        reurn : force of 3 x 22 x P
        """
        self.C70
        raise NotImplementedError()

    @Projector.pipeline
    def f_inv(self, forces, coords):
        """
         positions : 3 x A x N
         forces : 3 x A x N
         return 2 x N
        """
        mask, n = [4, 6, 14, 16], 2
        # mask, n = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21], 8
        c, phi_ax_idx, psi_ax_idx = (8, 6, 14)
        x = coords.copy()
        center = x[:, c]
        _p = x[:, :] - center[:, newaxis]
        phi_ax = _p[:, phi_ax_idx]
        psi_ax = _p[:, psi_ax_idx]

        p = _p[:, mask]
        lpl = norm(p, axis=0)
        e_p = p / lpl
        hax = repeat(phi_ax[:, newaxis], 2, axis=1)
        sax = repeat(psi_ax[:, newaxis], 2, axis=1)
        ax = concatenate((hax, sax), axis=1)
        laxl = norm(ax, axis=0)
        e_ax = ax / laxl

        r = lpl * (e_p - sum((e_p * e_ax), axis=0) * e_ax)
        f = forces[:, mask].copy()
        torque = cross(r, f, axis=0)
        thi = sum(torque[:, :n].sum(axis=1) * e_ax[:, 0], axis=0)
        tsi = sum(torque[:, n:].sum(axis=1) * e_ax[:, -1], axis=0)
        return np.array([thi, tsi]), self._x_inv(coords)


    def found_new_data(self, imgdata):
        count = imgdata._c.count()
        if count == 0:
            raise NotImplementedError('No data found')
        is_count_changed = self._cache['count'] != count
        return is_count_changed

    def get_inertia(self, positions, masses):
        pos = positions.copy()
        pos -= positions[8]
        ephi = pos[6] / norm(pos[6])
        epsi = pos[14] / norm(pos[14])
        phi = pos[:8]   # 8 x 3
        psi = pos[14:]  # 8 x 3
        I_phi = masses[:8] * norm(np.cross(phi, ephi), axis=1) ** 2   # 8
        I_psi = masses[14:] * norm(np.cross(psi, epsi), axis=1) ** 2  # 8
        return I_phi.sum(), I_psi.sum()

    def get_dihedral(self, p, a, b, c, d):
        # TODO: Merge at utils.py
        # TODO: Unit add
        # vector 1->2, 2->3, 3->4 and their normalized cross products:
        # p : 3 x N x P
        v_a = p[:, b, :] - p[:, a, :]  # 3 x P
        v_b = p[:, c, :] - p[:, b, :]
        v_c = p[:, d, :] - p[:, c, :]

        bxa = np.cross(v_b.T, v_a.T).T  # 3 x P
        cxb = np.cross(v_c.T, v_b.T).T
        bxanorm = np.linalg.norm(bxa, axis=0)  # P
        cxbnorm = np.linalg.norm(cxb, axis=0)
        if np.any(bxanorm == 0) or np.any(cxbnorm == 0):
            raise ZeroDivisionError('Undefined dihedral angle')
        bxa /= bxanorm  # 3 x P
        cxb /= cxbnorm
        angle = np.sum(bxa * cxb, axis=0)  # P
        # check for numerical trouble due to finite precision:
        angle[angle < -1] = -1
        angle[angle > 1] = 1
        angle = np.arccos(angle) * 180 / np.pi
        reverse = np.sum(bxa * v_c, axis=0) > 0
        angle[reverse] = 360 - angle[reverse]
        return angle.reshape(1, -1)

    def rotate(self, ref, a=None, v=None, i=8, mask=None):
        """Rotate atoms based on a vector and an angle, or two vectors.
        Parameters:
         ref  : C70 reference; C70; 3 x A x N  or  3 x A
         a    : angles   N
         v    : tuple, rotation axis index
         i    : center index ; 8
        Return:
         c : 3 x A x N  or  3 x A x 1
        """
        c = ref.copy()                 # 3 x A x N  or 3xA
        a = a.copy()
        v_i, v_f = v
        v = (c[:, v_i] - c[:, v_f])    # 3 x N or 3
        normv = norm(v, axis=0)        # N or 1

        if np.any(normv == 0.0):
            raise ZeroDivisionError('Cannot rotate: norm(v) == 0')

        # a *= np.pi / 180
        v /= normv                            # 3 x N  or  3
        co = cos(a)
        si = sin(a)                           # N  or  1
        center = c[:, i]                      # 3 x N     or  3
        c = c - center[:, np.newaxis]         # 3 x A x N or 3 x A

        # 3 x A x N @ 3 x 1 x N  ->  A x N
        c0v = sum(c * v[:, np.newaxis], axis=0)
        # 3 x A x N
        r = c * co - cross(c, (v * si), axis=0)
        # 3 x 1 x N (x) 1 x A x N -> 3 x A x N
        r = r + ((1.0 - co) * v)[:, np.newaxis] * c0v[np.newaxis, :]
        r = r + center[:, np.newaxis]
        return r[:, mask]

    def overlap_handler(self, coords):
        """
        Handles the too close atom # 17 - 5 distance issue:
        Data from ase.data covalent radii
        """
        c = coords.copy()
        cov_radii = (0.31 + 0.66)
        Hyd = c[:, 5, :]         # 3 x P
        Oxy = c[:, 17, :]        # 3 x P
        vec = Hyd - Oxy                     # 3 x P
        d = norm(vec, axis=0)               # P
        e_v = vec / d                       # 3 x P
        too_short = d < cov_radii * 0.7
        push = e_v[:, too_short] * cov_radii * 0.7 / 2
        c[:, 5, too_short] = Hyd[:, too_short] - push
        c[:, 17, too_short] = Oxy[:, too_short] + push
        return c
