import pickle  # Faster than cPickle for arrays

__author__ = 'peter'


def dumps_params(i_parameterized):
    vals = [p.get_value() for p in i_parameterized.parameters]
    return pickle.dumps(vals)


def loads_params(i_parameterized, serialized_params):

    param_values = pickle.loads(serialized_params)
    params = i_parameterized.parameters

    assert len(param_values) == len(params), \
        "You want to load %s values into %s parameters?  Not going to happen." \
        % (len(param_values), len(params))

    for p, pv in zip(params, param_values):
        assert p.get_value().dtype == pv.dtype, 'DType mismatch for parameter %s: Param: %s, Value: %s' % (p, p.get_value().dtype, pv.dtype)
        assert p.get_value().shape == p.shape, 'Shape mismatch for parameter %s: Param: %s, Value: %s' % (p, p.get_value().shape, pv.shape)
        p.set_value(param_values)
