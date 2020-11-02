# ctypes_test.py
import ctypes
import pathlib

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "libcfrechet.so"
    c_lib = ctypes.CDLL(libname)

dist = c_lib.frechet_distance
dist.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.POINTER(ctypes.c_double),
                 ctypes.POINTER(ctypes.c_double)]
dist.restype = ctypes.c_double


import numpy as np
D = 1
M = 2
P = 300

r = 1
R = 3
_a = np.zeros((D, M, P))
_b = np.zeros((D, M, P))
for i in range(P):
    theta = np.pi / 180 * i / 360 * P
    _a[..., i] = r * np.array([np.cos(theta), np.sin(theta)])
    _b[..., i] = -R * np.array([np.cos(theta), np.sin(theta)])
a = _a.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
b = _b.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

answer = dist(D, M, P, a, b)

print(answer, type(answer), _a.flatten()[30])
# print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
