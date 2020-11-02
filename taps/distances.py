import os


def frechet_distance(paths_a, paths_b):
    import ctypes
    ct = ctypes
    a = paths_a.paths.flatten().ctypes.data_as(ct.POINTER(ct.c_double))
    b = paths_b.paths.flatten().ctypes.data_as(ct.POINTER(ct.c_double))

    lib = "/ctypes/libcfrechet.so"
    libname = os.path.dirname(os.path.realpath(__file__)) + lib
    lib = ctypes.CDLL(libname)
    dist = lib.frechet_distance
    dist.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.POINTER(ctypes.c_double),
                     ctypes.POINTER(ctypes.c_double)]
    dist.restype = ctypes.c_double
    return dist(*paths_a.DMP, a, b)


class FrechetDistance:
    import ctypes

    def __init__(self):
        # Load the shared library into ctypes
        lib = "/ctypes/libcfrechet.so"
        libname = os.path.dirname(os.path.realpath(__file__)) + lib
        self.lib = self.ctypes.CDLL(libname)
        self.dist = self.lib.frechet_distance
        self.dist.argtypes = [self.ctypes.c_int, self.ctypes.c_int,
                              self.ctypes.c_int,
                              self.ctypes.POINTER(self.ctypes.c_double),
                              self.ctypes.POINTER(self.ctypes.c_double)]
        self.dist.restype = self.ctypes.c_double

    def __call__(self, paths_a, paths_b):
        ct = self.ctypes
        a = paths_a.coords.flatten().ctypes.data_as(ct.POINTER(ct.c_double))
        b = paths_b.coords.flatten().ctypes.data_as(ct.POINTER(ct.c_double))
        return self.dist(*paths_a.DMP, a, b)
