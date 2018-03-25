import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import keras.layers as KL
from keras import activations, regularizers, constraints
import keras.backend as K
import keras.regularizers as KR
from keras.utils import conv_utils
# from .cpx_init import ComplexInit, ComplexIndependentFilters
import cpx_initializers as cinitializers


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    # condition中为Ture的位置,输出值为small_res的值,false时则为large_res的值
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, k_h=5, k_w=5, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
    return bn


def deconv2d(input, deconv_info, is_train, name="deconv2d", stddev=0.02, activation_fn=None):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        deconv = layers.conv2d_transpose(
            input, num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            kernel_size=[k, k], stride=[s, s], padding='VALID'
        )
        if not activation_fn:
            deconv = tf.nn.relu(deconv)
            deconv = tf.contrib.layers.batch_norm(
                deconv, center=True, scale=True,  decay=0.9,
                is_training=is_train, updates_collections=None
            )
        else:
            deconv = activation_fn(deconv)
        return deconv


def learnConcatRealImagBlock(I, filter_size, featmaps, stage, block, bnArgs, init_act):
    """Learn initial imaginary component for input."""

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    O = KL.BatchNormalization(name=bn_name_base + '2a', **bnArgs)(I)
    O = KL.Activation(init_act)(O)
    O = KL.Convolution2D(featmaps[0], filter_size,
                      name=conv_name_base + '2a',
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=KR.l2(0.0001))(O)

    O = KL.BatchNormalization(name=bn_name_base + '2b', **bnArgs)(O)
    O = KL.Activation(init_act)(O)
    O = KL.Convolution2D(featmaps[1], filter_size,
                      name=conv_name_base + '2b',
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=KR.l2(0.0001))(O)

    return O


def flatten_fully(inputs):
    input_shape = K.int_shape(inputs)
    input_dim = input_shape[-1]//2
    real_input = inputs[:, :, :, :input_dim]
    imag_input = inputs[:, :, :, input_dim:]
    copx_input = tf.complex(real_input, imag_input)
    copx_shape = K.int_shape(copx_input)
    dim_flat = np.prod(copx_shape[1:])
    copx_input = tf.reshape(copx_input, tf.stack([-1, dim_flat]))
    real_input = tf.real(copx_input)
    imag_input = tf.imag(copx_input)
    outputs = KL.Concatenate(-1)([real_input, imag_input])
    return outputs

# def sanitizedInitGet(init):
#     if init in ["sqrt_init"]:
#         return sqrt_init
#     elif init in ["complex", "complex_independent",
# 	              "glorot_complex", "he_complex"]:
#         return init
#     else:
# 		return initializers.get(init)


# def sanitizedInitSer(init):
# 	if init in [sqrt_init]:
# 		return "sqrt_init"
# 	elif init == "complex" or isinstance(init, ComplexInit):
# 		return "complex"
# 	elif init == "complex_independent" or isinstance(init, ComplexIndependentFilters):
# 		return "complex_independent"
# 	else:
# 		return initializers.serialize(init)


# def sqrt_init(shape, dtype=None):
#     value = (1 / K.sqrt(2)) * K.ones(shape)
#     return value


class _ComplexConv(KL.Layer):
    # 论文源代码中这个地方外加了两个功能，都是针对权值的
    #   1、spectral parametrization: 表示权值是在谱域定义的，进行卷积之前需要把其用IFFT转变为原始的空域
    #   2、normalize weight: 每一次进行卷积之前都把权值归一化，采用的是bn里面定义的complex_normalization，这也应该是在complexBN中作者没有把那两个函数放到类里面的原因吧
    # 关于1，和pooling里面的那个SpectralPooling均是采用论文：Spectral Representations for Convolutional Neural Networks（https://arxiv.org/abs/1506.03767）
    #       卷积的权值在频率域表示，进行卷积之前需要用IFFT把其反变换为空域，论文中显示，这种表示方法学习到的权值更稀疏更少（学习改变的权值的量少）
    # 关于2，个人感觉没什么用，在这个地方直接对权值进行操作不如加相似功能的正则化
    #
    # 卷积网络里面越倾向高层，卷积层输出得到的特征谱越稀疏
    # 保留1，舍弃2
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides = 1,
                 padding = 'valid',
                 data_format = None,
                 dilation_rate = 1,
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'he_complex',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 seed = None,
                 spectral_parametrization = False,
                 **kwargs):
        super(_ComplexConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias  = use_bias
        self.kernel_initializer = cinitializers.get(kernel_initializer)
        self.bias_initializer = cinitializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = KL.InputSpec(ndim = self.rank + 2)
        self.spectral_parametrization = spectral_parametrization
        self.seed = seed if seed is not None else np.random.randint(1, 1e6)

    def build(self, input_shape):
        # input_shapes = input_shape
        # assert (input_shapes[0] == input_shapes[1])
        # input_shape = input_shapes[0]
        # assert len(input_shape) >= 2

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]//2
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel_shape = kernel_shape
        self.kernel = self.add_weight(shape = kernel_shape,
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint)
        self.kernel_real = self.kernel[0]
        self.kernel_image = self.kernel[1]
        self.kernel_complex = tf.complex(self.kernel_real, self.kernel_image)

        if self.use_bias:
            self.bias_real = self.add_weight(shape = (self.filters,),
                                             initializer = self.bias_initializer,
                                             name = 'bias_real',
                                             regularizer = self.bias_regularizer,
                                             constraint = self.bias_constraint)
            self.bias_image = self.add_weight(shape = (self.filters,),
                                              initializer = self.bias_initializer,
                                              name = 'bias_image',
                                              regularizer = self.bias_regularizer,
                                              constraint = self.bias_constraint)
        else:
            self.bias_real = None
            self.bias_image = None

        self.input_spec = KL.InputSpec(ndim = self.rank + 2,
                                    axes = {channel_axis: input_dim*2})
        self.built = True

        self.convArgs = {
            'strides':          self.strides[0] if self.rank ==1 else self.strides,
            'padding':          self.padding,
            'data_format':      self.data_format,
            'dilation_rate':    self.dilation_rate[0] if self.rank == 1 else self.dilation_rate
        }
        self.convFunc = {
            1:  K.conv1d,
            2:  K.conv2d,
            3:  K.conv3d
        }[self.rank]
        self.ifftFunc = {
            1:  tf.ifft,
            2:  tf.ifft2d,
            3:  tf.ifft3d
        }[self.rank]

    def call(self, inputs, **kwargs):
        # assert isinstance(inputs, list)

        input_shape = K.int_shape(inputs)
        num_channel = input_shape[3]//2
        input_real, input_image = inputs[:, :, :, :num_channel], inputs[:, :, :, num_channel:]
        kernel_complex = self.kernel_complex
        # transform weights to spectral domain
        if self.spectral_parametrization:
            flat_shape = (self.kernel_shape[-1] * self.kernel_shape[-2],) + self.kernel_shape[:-2]
            fk = K.reshape(kernel_complex, flat_shape)
            fk = self.ifftFunc(fk)
            kernel_complex = K.reshape(fk, self.kernel_shape)
        kernel_real = tf.real(kernel_complex)
        kernel_image = tf.imag(kernel_complex)

        output_real = self.convFunc(input_real, kernel_real, **self.convArgs) - self.convFunc(input_image, kernel_image, **self.convArgs)
        output_image = self.convFunc(input_real, kernel_image, **self.convArgs) + self.convFunc(input_image, kernel_real, **self.convArgs)

        if self.use_bias:
            output_real = K.bias_add(output_real, self.bias_real)
            output_image = K.bias_add(output_image, self.bias_image)
        if self.activation is not None:
            output_real = self.activation(output_real)
            output_image = self.activation(output_image)
        output = K.concatenate([output_real, output_image], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        # input_shapes = input_shape
        # assert (input_shapes[0] == input_shapes[1])
        # input_shape = input_shapes[0]
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            single_shape = (input_shape[0],) + tuple(new_space) + (2*self.filters,)
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            single_shape = (input_shape[0], 2*self.filters) + tuple(new_space)
        else:
            raise ValueError('Invalid data format: ' + self.data_format)
        return single_shape

    def get_config(self):
        config = {
            'rank':                 self.rank,
            'filters':              self.filters,
            'kernel_size':          self.kernel_size,
            'strides':              self.strides,
            'padding':              self.padding,
            'data_format':          self.data_format,
            'dilation_rate':        self.dilation_rate,
            'activation':           activations.serialize(self.activation),
            'use_bias':             self.use_bias,
            'kernel_initializer':   cinitializers.serialize(self.kernel_initializer),
            'bias_initializer':     cinitializers.serialize(self.bias_initializer),
            'kernel_regularizer':   regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':     regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':    constraints.serialize(self.kernel_constraint),
            'bias_constraint':      constraints.serialize(self.bias_constraint),
            'spectral_parametrization': self.spectral_parametrization,
            'seed':                     self.seed,
        }
        base_config = super(_ComplexConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexConv2D(_ComplexConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides = (1, 1),
                 padding = 'valid',
                 dilation_rate = (1, 1),
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'he_complex',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 spectral_parametrization = False,
                 **kwargs):
        super(ComplexConv2D, self).__init__(
            rank = 2,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            dilation_rate = dilation_rate,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer,
            kernel_constraint = kernel_constraint,
            bias_constraint = bias_constraint,
            spectral_parametrization = spectral_parametrization,
            **kwargs
        )
