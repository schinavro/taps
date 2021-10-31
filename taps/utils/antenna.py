import json
import warnings
import inspect
import numpy as np
from collections import deque
from importlib import import_module

from ase.atoms import Atoms

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
    header = np.array([pointer, typemaps[dtype], ndim, *shape,
                       ordermaps[order]], dtype=np.int64)
    headerbytes = header.tobytes()
    return headerbytes


def read_header(arrbytes):
    header_size = int.from_bytes(arrbytes[:8], 'little', signed=True)

    header = np.frombuffer(arrbytes[8:8+8*header_size], dtype=np.int64,
                           count=header_size//8)
    pointer, dtype, ndim = header[0], typemaps[header[1]], header[2]
    shape = header[3:3+ndim]
    order = ordermaps[header[3+ndim]]
    return header_size, pointer, dtype, ndim, shape, order


def dictify(obj, ignore_cache=True):
    if isinstance(obj, type) or inspect.ismethod(obj):
        return None
    elif isinstance(obj, dict):
        dct = {}
        for k, v in obj.items():
            if isinstance(k, int):
                pass
            elif isinstance(k, bytes):
                k = '___' + k.hex()
            elif (k[0] == '_' and ignore_cache):
                continue

            if v is None or isinstance(v, type) or inspect.ismethod(v):
                continue
            elif isinstance(v, (dict, list)):
                dct[k] = dictify(v, ignore_cache=ignore_cache)
            elif v.__class__.__module__ in ['builtins', 'numpy']:
                dct[k] = v
            else:
                dct[k] = dictify(v, ignore_cache=ignore_cache)
        return dct

    elif isinstance(obj, list):
        lst = []
        for v in obj:
            if v is None or isinstance(v, type) or inspect.ismethod(v):
                continue
            elif isinstance(v, (dict, list)):
                lst.append(dictify(v, ignore_cache=ignore_cache))
            elif v.__class__.__module__ in ['builtins', 'numpy']:
                lst.append(v)
            else:
                lst.append(dictify(v, ignore_cache=ignore_cache))
        return lst
    elif isinstance(obj, Atoms):
        from ase.db.row import atoms2dict
        kwargs = atoms2dict(obj)
        kwargs['cell'] = kwargs['cell'].array
        return {'__name__': 'Atoms2',
                '__module__': obj.__class__.__module__,
                'kwargs': kwargs}
    else:
        dct = {}
        for k, v in vars(obj).items():
            if isinstance(k, int):
                pass
            elif isinstance(k, bytes):
                k = '___' + k.hex()
            elif (k[0] == '_') and ignore_cache:
                continue

            if v is None or isinstance(v, type) or inspect.ismethod(v):
                continue
            elif isinstance(v, (dict, list)):
                dct[k] = dictify(v, ignore_cache=ignore_cache)
            elif v.__class__.__module__ in ['builtins', 'numpy']:
                dct[k] = v
            else:
                dct[k] = dictify(v, ignore_cache=ignore_cache)
        return {'__name__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__, 'kwargs': dct}


def classify(obj):
    if isinstance(obj, dict) and obj.get('__name__') == 'Atoms2':
        from ase.db.row import AtomsRow
        return AtomsRow(obj['kwargs']).toatoms(attach_calculator=True,
                                      add_additional_information=True)
    elif isinstance(obj, dict) and obj.get('__module__') is not None:
        module = import_module(obj['__module__'])
        ##
        sig = inspect.signature(getattr(module, obj['__name__']).__init__)
        init_keys = sig.parameters.keys()
        obj_keys = list(obj['kwargs'].keys())
        init_values = sig.parameters.values()
        has_kwargs = any([True for p in init_values if p.kind == p.VAR_KEYWORD])
        if has_kwargs:
            pass
        else:
            for k in obj_keys:
                if k not in init_keys:
                    warnings.warn('key  %s not in %s' % (k, obj['__name__']))
                    del obj['kwargs'][k]
        return getattr(module, obj['__name__'])(**(classify(obj['kwargs'])))
    elif isinstance(obj, list):
        lst = []
        for v in obj:
            if isinstance(v, (list, dict)):
                lst.append(classify(v))
            elif v.__class__.__module__ in ['builtins', 'numpy']:
                lst.append(v)
            else:
                typename = v.__class__.__name__
                raise NotImplementedError("Type invalid %s" % typename)
        return lst
    elif isinstance(obj, dict):
        kwargs = {}
        for k, v in obj.items():
            if len(k) > 3 and (k[:3] == '___'):
                k = bytes.fromhex(k[3:])
            if isinstance(v, (list, dict)):
                kwargs[k] = classify(v)
            elif v.__class__.__module__ in ['builtins', 'numpy']:
                kwargs[k] = v
            else:
                typename = (k, v.__class__)
                raise NotImplementedError("Type invalid %s %s" % typename)
        return kwargs
    else:
        raise NotImplementedError("Invalid type %s" % obj.__class__.__name__)


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
                    arr = arrlist[i][1]
                    if arr.dtype == np.uint8:
                        arr = arr.tobytes()
                    o[k] = arr
        elif isinstance(o, (list, tuple)):
            for i in range(len(o)):
                v = o[i]
                queue.append((id(v), v))
                if v in pointerlist:
                    j = pointerlist.index(v)
                    arr = pointerlist.index[j]
                    if arr.dtype == np.uint8:
                        arr = arr.tobytes()
                    o[i] = arr
    return d


def pointify(d, pointer=0, binarylist=[]):
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
                if np.issubdtype(type(v), np.integer):
                    o[k] = int(v)
                elif np.issubdtype(type(v), np.floating):
                    o[k] = float(v)
                elif np.issubdtype(type(v), np.str_):
                    o[k] = str(v)
                elif np.issubdtype(type(v), np.complexfloating):
                    o[k] = complex(v)
                if isinstance(v, (np.ndarray, bytes)):
                    vv = v.copy()
                    if isinstance(vv, bytes):
                        vv = np.frombuffer(vv, dtype=np.uint8)
                    pstr = "__%d__" % pointer
                    headerbytes = write_header(pointer, vv)
                    hsize = int.to_bytes(len(headerbytes), 8, 'little')
                    binarylist.append(hsize + headerbytes + vv.tobytes())
                    o[k] = pstr
                    pointer += 1



        elif isinstance(o, (list, tuple)):
            for i in range(len(o)):
                v = o[i]
                queue.append((id(v), v))
                if isinstance(v, (np.ndarray, bytes)):
                    vv = v.copy()
                    if isinstance(vv, bytes):
                        vv = np.frombuffer(vv, dtype=np.uint8)
                    pstr = '__%d__' % pointer
                    headerbytes = write_header(pointer, vv)
                    hsize = int.to_bytes(len(headerbytes), 8, 'little')
                    binarylist.append(hsize + headerbytes + vv.tobytes())
                    o[i] = pstr
                    pointer += 1
                elif isinstance(v, np.int64):
                    o[k] = int(v)

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


def unpacking(bytesarr, includesize=False):
    if includesize:
        bytesarr = bytesarr[8:]
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
