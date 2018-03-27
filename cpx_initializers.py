#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2017/12/28 22:06
# Author    : zsh_o

from keras import backend as K
from keras import initializers
from keras.initializers import Initializer
import numpy as np
from numpy.random import RandomState
import six
from keras.utils.generic_utils import serialize_keras_object,deserialize_keras_object

class Independent(Initializer):
    # Make every filters different from each other
    # The number of filters: Input_dim * Output_dim
    # The size of Dense filter is 1-dim
    # The size of ConvND filters is N-dim

    def __init__(self, flattened = False,criterion = 'glorot', seed = None):
        # flattened = True: used for Dense
        # flattened = False: used for multi-dim filter

        self.flattened = flattened
        self.criterion = criterion
        self.seed = 2345 if seed is None else seed

    def __call__(self, shape, dtype = None):
        if self.flattened is True:
            # Dense
            num_rows = np.prod(shape)
            num_cols = 1
            fan_in = np.prod(shape[:-1])
        else:
            # Conv
            num_rows = np.prod(shape[-2:])
            num_cols = np.prod(shape[:-2])
            fan_in = shape[-2]
        fan_out = shape[-1]

        flat_shape = (num_rows, num_cols)
        rng = RandomState(self.seed)
        x = rng.uniform(size = flat_shape)
        u, _, v = np.linalg.svd(x)
        orthogonal_x = np.dot(u, np.dot(np.eye(flat_shape), v.T))
        independent_filters = np.reshape(orthogonal_x, shape)

        if self.criterion == 'glorot':
            desired_var = 2. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 2. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_constant = np.sqrt(desired_var / np.var(independent_filters))
        weight = multip_constant * independent_filters
        return weight

    def get_config(self):
        return {
            'flattened': self.flattened,
            'criterion': self.criterion,
            'seed': self.seed
        }

class ComplexIndependent(Initializer):
    # Make every filters different from each other
    # The number of filters: Input_dim * Output_dim
    # The size of Dense filter is 1-dim
    # The size of ConvND filters is N-dim

    def __init__(self, flattened = False,criterion = 'glorot', seed = None):
        # flattened = True: used for Dense
        # flattened = False: used for multi-dim filter

        self.flattened = flattened
        self.criterion = criterion
        self.seed = 2345 if seed is None else seed

    def __call__(self, shape, dtype = None):
        if self.flattened is True:
            # Dense
            num_rows = np.prod(shape)
            num_cols = 1
            fan_in = np.prod(shape[:-1])
        else:
            # Conv
            num_rows = np.prod(shape[-2:])
            num_cols = np.prod(shape[:-2])
            fan_in = shape[-2]
        fan_out = shape[-1]

        flat_shape = (int(num_rows), int(num_cols))
        rng = RandomState(self.seed)
        r = rng.uniform(size = flat_shape)
        i = rng.uniform(size = flat_shape)
        z = r + 1j*i
        u, _, v = np.linalg.svd(z)
        unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
        independent_filters = np.reshape(unitary_z, shape)
        indep_real = independent_filters.real
        indep_image = independent_filters.imag

        if self.criterion == 'glorot':
            desired_var = 2. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 2. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_real = np.sqrt(desired_var / np.var(indep_real))
        weight_real = multip_real * indep_real
        multip_image = np.sqrt(desired_var / np.var(indep_image))
        weight_image = multip_real * indep_image

        # weight = np.concatenate([weight_real, weight_image], axis = -1)
        # 为了便于能将weight_real 和 weight_image 分割开，采用stack, 而不是concatenate
        # weight_real = weight[0], weight_image = weight[1]
        weight = np.stack([weight_real, weight_image])
        return weight

    def get_config(self):
        return {
            'flattened': self.flattened,
            'criterion': self.criterion,
            'seed': self.seed
        }

class ComplexInit(Initializer):
    # Generate complex weights by moduls and phase
    # Moduls from rayleigh distribution with s = 1/(fan_in + fan_out) if criterion == 'glorot' else s = 1/(fan_in) if criterion == 'he'
    # phase from uniform distribution with low = -pi, high = pi

    def __init__(self, flattened = False,criterion = 'glorot', seed = None):
        # flattened = True: used for Dense
        # flattened = False: used for multi-dim filter

        self.flattened = flattened
        self.criterion = criterion
        self.seed = 2345 if seed is None else seed

    def __call__(self, shape, dtype=None):
        fan_in = np.prod(shape[:-1]) if self.flattened is True else shape[-2]
        fan_out = shape[-1]

        if self.criterion == 'glorot':
            s = 2. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 2. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        rng = RandomState(self.seed)
        modulus = rng.rayleigh(scale = s, size = shape)
        phase = rng.uniform(low = -np.pi, high = np.pi, size = shape)
        weight_real = modulus * np.cos(phase)
        weight_image = modulus * np.sin(phase)
        # weight = np.concatenate([weight_real, weight_image], axis = -1)
        weight = np.stack([weight_real, weight_image])
        return weight

class SqrtInit(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1 / K.sqrt(2), shape=shape, dtype=dtype)

# keras 可接受的initializer均是可运行的函数（或者定义了__call__的类）并且参数只能是shape，dtype，所以带其他初始化参数的initializer类需要预先初始化
glorot_independent = Independent(criterion = 'glorot')
he_independent = Independent(criterion = 'he')
glorot_complex_independent = ComplexIndependent(criterion = 'glorot')
he_complex_independent = ComplexIndependent(criterion = 'he')
glorot_complex = ComplexInit(criterion = 'glorot')
he_complex = ComplexInit(criterion = 'he')
sqrt = SqrtInit

def serialize(initializer):
    return serialize_keras_object(initializer)

def deserialize(config, custom_objects = None):
    return deserialize_keras_object(config,
                                    module_objects = globals(),
                                    custom_objects = custom_objects,
                                    printable_module_name = 'complex_initializer')

# initializer不在该作用域范围内，则转到keras官方initializers类
def get(identifier):
    module_lists = globals()
    if isinstance(identifier, dict):
        class_name = identifier['class_name']
        if module_lists.get(class_name) is None:
            return initializers.get(identifier)
        else:
            return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        if module_lists.get(identifier) is None:
            return initializers.get(identifier)
        else:
            config = {'class_name': str(identifier), 'config': {}}
            return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier:',
                         identifier)