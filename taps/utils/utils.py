import numpy as np
from functools import wraps
from collections import OrderedDict
from ase.calculators.calculator import get_calculator_class, Parameters


class Images:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, index=np.s_[:]):
        if np.isscalar(index):
            index = [index]
        return self.get_images(index)

    def __setitem__(self, index=np.s_[:], value=None):
        if np.isscalar(index):
            index = [index]
        self.set_images(index, value)

    def __iter__(self):
        for i in self[np.s_[:]]:
            yield i


class ImageIndexing:
    def __init__(self, get_images=None, name=None):
        self.Images__dict__ = {}
        self.construct_method('get_images', get_images)
        self.construct_class()

    def construct_class(self):
        self.Images = type(self.__name__, (Images,), self.Images__dict__)

    def __get__(self, obj, objtype=None):
        return self.Images(obj)

    def __set__(self, obj, value):
        raise AttributeError("can't assign to " + type(self).__name__)

    def construct_method(self, name, fn):
        @wraps(fn)
        def wrapper(Images, *args, **kwargs):
            return fn(Images.obj, *args, **kwargs)
        self.Images__dict__[name] = wrapper

        for attr in ('__module__', '__name__', '__qualname__',
                     '__annotations__', '__doc__'):
            setattr(self, attr, getattr(fn, attr))

    def getter(self, fn):
        self.construct_method('get_images', fn)
        self.construct_class()
        return self

    def setter(self, fn):
        self.construct_method('set_images', fn)
        self.construct_class()
        return self


def paths2dct(paths, necessary_parameters, allowed_properties, class_objects,
              save_calc=False, save_model=False, save_finder=False,
              save_prj=False, save_imgdata=False, save_plotter=False):
    dct = OrderedDict()
    P, N = paths.P, paths.N
    for key, value in paths.__dict__.items():
        if key[0] == '_':
            continue
        if key in necessary_parameters.keys():
            default = necessary_parameters[key]['default']
            isDefault = np.all(default.format(P=P, N=N) == value)
            if not isDefault:
                dct[key] = value
        elif key == 'calc' and save_calc:
            dct[key] = value.name
            calc_parameters = []
            calc_results = []
            for calc in value:
                calc_parameters.append(calc.parameters)
                calc_results.append(calc.results)
            dct['calc_parameters'] = calc_parameters
            dct['calc_results'] = calc_results
        elif key == 'constraints':
            constraints = []
            for const in value:
                constraints.append([c.todict() for c in const])
            dct[key] = constraints
        elif key == 'info':
            info = []
            for inf in value:
                info.append(inf)
            dct[key] = info
        elif key in allowed_properties.keys():
            properties = []
            call = allowed_properties[key]['call'].format(atoms='atoms')
            for atoms in paths.images:
                properties.append(eval(call).tolist())
            dct[key] = properties
        else:
            NotImplementedError('This item, %s, seems from nowhere' % key)

    for key, value in paths.invariants.items():
        if key == 'calc' and save_calc:
            dct[key] = value.name
            dct['calc_parameters'] = value.parameters
            dct['calc_results'] = value.results

        elif key == 'constraints':
            dct[key] = [c.todict() for c in value]
        else:
            dct[key] = value

    for object in class_objects.keys():
        if not locals()['save_' + object]:
            continue
        self_object = getattr(paths, object)
        dct[object] = self_object.__class__.__name__
        object_parameters = getattr(self_object, object + '_parameters')
        for key in object_parameters.keys():
            if key[0] == '_':
                continue
            value = getattr(self_object, key, None)
            if value is None:
                continue
            if value.__class__.__module__ in ['builtins', 'numpy',
                                              'collections']:
                pass
            else:
                value = value.__class__.__name__
            default = object_parameters[key]['default']
            asse = object_parameters[key]['assert']
            assert eval(asse.format(name='value')), '%s %s' % (key, value)
            if np.any(eval(default) != value):
                dct[object + '_' + key] = value

    return paths.symbols, paths.coords.copy(), dct


def dct2pd_dct(dct):
    pd_dct = {}
    for key, value in dct.items():
        if value.__class__ in [dict, list, tuple, Parameters, OrderedDict]:
            # if len(value) != 1:
            value = [value]
        elif value.__class__ in [np.ndarray]:
            value = [value.tolist()]
        pd_dct[key] = value
    return pd_dct


def pandas2dct(label, index=None, return_dataframe=False):
    import pandas as pd
    try:
        filename, format = label.rsplit('.', 1)
        if format not in ['csv', 'pkl']:
            filename = label
            format = 'csv'
    except ValueError:
        filename = label
        format = 'csv'
    if format == 'csv' or format is None:
        dataframe = pd.read_csv(filename + '.csv')
        reader = csvreader
    elif format == 'pkl':
        dataframe = pd.read_pickle(filename + '.' + format)
        reader = pklreader
    else:
        NotImplementedError('Format %s not support' % format)
    if return_dataframe:
        return dataframe

    return dataframe2dct(dataframe, index=index, reader=reader)


def pklreader(value):
    if value.__class__ in [list, np.ndarray]:
        return np.array(value)
    else:
        return value


def csvreader(value):
    if value.__class__ == str:
        if value[0] == '[':
            return np.array(eval(value))
        elif value[0] == '(':
            return eval(value)
        elif value[0] == '{':
            return eval(value)
        elif value[0] == 'O':
            # OrderedDict
            return eval(value)
        elif value.lower() in ['true', 'false']:
            return value[0] == 'T'
        else:
            return value
    else:
        return value


def dfseries2dct(series, reader=csvreader):
    def df(key):
        return reader(series[key])
    symbols = df('symbols')
    paths = np.array(df('coords'))
    kwargs = {}

    for key, value in series.items():
        if key in ['symbols', 'coords', 'calc_parameters', 'calc_results']:
            continue
        elif 'Unnamed' in key:
            continue
        if key in ['calc']:
            calculators = df(key).lower()
            parameters = df('calc_parameters')
            calc = get_calculator_class(calculators)(**parameters)
            results = df('calc_results')
            calc.results = results
            kwargs[key] = calc
        elif key == 'constraints':
            raise NotImplementedError('Wait')
        else:
            kwargs[key] = df(key)
    return symbols, paths, kwargs


def dataframe2dct(dataframe, index=None, reader=csvreader):
    def df(key):
        return reader(dataframe[key][index])
    symbols = df('symbols')
    coords = np.array(df('coords'))
    kwargs = {}

    for key, value in dataframe.iloc[index].items():
        if key in ['symbols', 'coords', 'calc_parameters', 'calc_results']:
            continue
        elif 'Unnamed' in key:
            continue
        # elif 'finder_res' == key:
        #    continue
        # elif 'imgdata_database' == key:
        #    continue
        if key in ['calc']:
            calculators = df(key).lower()
            parameters = df('calc_parameters')
            calc = get_calculator_class(calculators)(**parameters)
            results = df('calc_results')
            calc.results = results
            kwargs[key] = calc
        elif key == 'constraints':
            raise NotImplementedError('Wait')
        else:
            kwargs[key] = df(key)
    return symbols, coords, kwargs


def wrap_coords(coords, cell):
    if len(cell.shape) == 1:
        origin = -cell[:, np.newaxis, np.newaxis] / 2
    elif len(cell.shape) == 2:
        origin = -cell[:, np.newaxis] / 2
    elif len(cell.shape) == 3:
        assert coords.shape[-1] == cell.shape[-1]
        origin = -cell / 2
    return (coords - origin) % cell + origin
