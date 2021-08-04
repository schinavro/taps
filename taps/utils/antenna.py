import json
import numpy as np
from collections import deque

Int, Float, Real, Number, Bool = 1, 2, 3, 4, 5
Int8, Int16, Int32, Int64 = 6, 7, 8, 9
UInt8, UInt16, UInt32, UInt64 = 10, 11, 12, 13
Float16, Float32, Float64 = 14, 15, 16
ComplexIFloat32I, ComplexIFloat64I = 17, 18

typemaps = {
    int: Int, Int: int,
    np.dtype('float'): Float, Float: 'float', Real: float, Number: float,
    np.dtype('bool'): Bool, Bool: np.bool,
    np.dtype('int8'): Int8, Int8: np.int8,
    np.dtype('int16'): Int16, Int16: np.int16,
    np.dtype('int32'): Int32, Int32: np.int32,
    np.dtype('int64'): Int64, Int64: np.int64,
    np.dtype('uint8'): UInt8, UInt8: np.uint8,
    np.dtype('uint16'): UInt16, UInt16: np.uint16,
    np.dtype('uint32'): UInt32, UInt32: np.uint32,
    np.dtype('uint64'): UInt64, UInt64: np.uint64,
    np.dtype('float16'): Float16, Float16: np.float16,
    np.dtype('float32'): Float32, Float32: np.float32,
    np.dtype('float64'): Float64, Float64: np.float64,
    np.dtype('complex64'): ComplexIFloat32I, ComplexIFloat32I: np.complex64,
    np.dtype('complex128'): ComplexIFloat64I, ComplexIFloat64I: np.complex128
    }

ordermaps = {"C": 0, 0: "C", 1: "F", "F": 1}


def write_header(pointer, arr):
    header = []
    dtype = arr.dtype
    ndim = arr.ndim
    shape = arr.shape
    order = "C"
    header = np.array([pointer, typemaps[dtype], ndim, *shape, ordermaps[order]], dtype=np.int64)
    headerbytes = header.tobytes()
    return headerbytes


def read_header(arrbytes):
    header_size = int.from_bytes(arrbytes[:8], 'little', signed=True)

    header = np.frombuffer(arrbytes[8:8+8*header_size], dtype=np.int64, count=header_size//8)
    pointer, dtype, ndim = header[0], typemaps[header[1]], header[2]
    shape = header[3:3+ndim]
    order = ordermaps[header[3+ndim]]
    return header_size, pointer, dtype, ndim, shape, order


def statify(arrlist, d):
    pointerlist = ['__%d__' % arr[0] for arr in arrlist]
    queue = deque([(id(d), d)])
    memo = set()
    while queue:
        id_, o = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            for k, v in o.items():
                queue.append((id(v), v))
                if v in pointerlist:
                    i = pointerlist.index(v)
                    o[k] = arrlist[i][1]
        elif isinstance(o, (list, tuple)):
            for i in range(len(o)):
                v = o[i]
                queue.append((id(v), v))
                if v in pointerlist:
                    j = pointerlist.index(v)
                    o[i] = pointerlist.index[j]
    return d


def pointify(d, pointer, binarylist):
    queue = deque([(id(d), d)])
    memo = set()
    while queue:
        id_, o = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            for k, v in o.items():
                queue.append((id(v), v))
                typename = v.__class__.__name__

                if typename == "ndarray":
                    pstr = "__%d__" % pointer
                    headerbytes = write_header(pointer, o[k])
                    binar = int.to_bytes(len(headerbytes), 8, 'little') + headerbytes + v.tobytes()
                    binarylist.append(binar)
                    o[k] = pstr
                    pointer += 1
        elif isinstance(o, (list, tuple)):
            for i in range(len(o)):
                v = o[i]
                queue.append((id(v), v))
                typename = v.__class__.__name__
                if typename == "ndarray":
                    pstr = '__%d__' % pointer
                    headerbytes = write_header(pointer, o[i])
                    binar = int.to_bytes(len(header), 8, 'little') + headerbytes + v.tobytes()
                    binarylist.append(binar)
                    o[i] = pstr
                    pointer += 1
    return d, pointer, binarylist


def packing(*args, **kwargs):
    binarylist = []
    pointer = 0

    args, pointer, binarylist = pointify(list(args), pointer, binarylist)
    kwargs, pointer, binarylist = pointify(kwargs, pointer, binarylist)
    kwargs["args"] = args

    howmanybinary = int.to_bytes(len(binarylist), 8, 'little')
    eachsize = np.array([len(b) for b in binarylist], dtype=np.int64).tobytes()
    binarybytes = howmanybinary + eachsize + b''.join(binarylist)
    kwargsbytes = json.dumps(kwargs).encode("utf-8")

    data = binarybytes + kwargsbytes
    header = np.array([len(data)+16, len(binarybytes), len(kwargsbytes)], dtype=np.int64).tobytes()
    return header + data


def unpacking(bytesarr):
    nargs, nkwargs = np.frombuffer(bytesarr[:16], dtype=np.int64, count=2)
    partition = 16 + nargs

    binarybytes = bytesarr[16: partition]
    kwargsbytes = bytesarr[partition:partition+nkwargs]

    # binarybytes to binarylist
    howmanybinary = int.from_bytes(binarybytes[:8], 'little', signed=True)

    eachsize = np.frombuffer(binarybytes[8:], dtype=np.int64, count=howmanybinary)
    partition = 8 + 8*howmanybinary

    arrlist = []

    for size in eachsize:
        arrbytes = binarybytes[partition: partition+size]
        header_size, pointer, dtype, ndim, shape, order = read_header(arrbytes)
        arr = np.frombuffer(arrbytes[header_size+8:], dtype=dtype)
        if order == "F":
            shape = np.flip(shape)
        array = arr.reshape(shape)
        arrlist.append((pointer, array))
        partition += size
    pkwargs = json.loads(kwargsbytes.decode('utf-8'))
    pargs = pkwargs["args"]
    del pkwargs["args"]

    # link binarybytes and everything
    args = statify(arrlist, pargs)
    kwargs = statify(arrlist, pkwargs)

    return args, kwargs