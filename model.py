from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from keras import backend as K
import keras.regularizers as KR
import keras.layers as KL
from ops import ComplexConv2D, deconv2d, huber_loss, learnConcatRealImagBlock, flatten_fully
from cpx_bn import ComplexBatchNormalization as complexBN
from cpx_dense import ComplexDense
from capsulelayers import PrimaryCap, CapsuleLayer, shape_loss
from util import log


class SGAN_Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.num_class = self.config.data_info[2]
        self.c_dim = self.config.data_info[3]
        self.deconv_info = self.config.deconv_info
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')
        # 对权重正则化时的系数
        self.recon_weight = tf.placeholder_with_default(
            tf.cast(1.0, tf.float32), [])
        tf.summary.scalar("loss/recon_wieght", self.recon_weight)

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'], # [B, h, w, c]
            self.label: batch_chunk['label'], # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        # Weight annealing
        if step is not None:
            fd[self.recon_weight] = min(max(0, (1500 - step) / 1500), 1.0)*10
        return fd

    def build(self, is_train=True):

        n = self.num_class
        deconv_info = self.deconv_info
        conv_info = self.conv_info
        n_z = 100

        # build loss and accuracy {{{
        def build_loss(d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image):
            alpha = 0.9
            real_label = tf.concat([label, tf.zeros([self.batch_size, 1])], axis=1)
            fake_label = tf.concat([(1-alpha)*K.ones([self.batch_size, n])/n, alpha*K.ones([self.batch_size, 1])], axis=1)

            # Discriminator/classifier loss
            s_loss = tf.reduce_mean(huber_loss(label, d_real[:, :-1]))
            d_loss_real = tf.nn.softmax_cross_entropy_with_logits(logits=d_real_logits, labels=real_label)
            d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=d_fake_logits, labels=fake_label)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

            # Generator loss
            g_loss = tf.reduce_mean(K.log(d_fake[:, -1]))

            # Weight annealing
            g_loss += tf.reduce_mean(huber_loss(real_image, fake_image)) * self.recon_weight

            GAN_loss = tf.reduce_mean(d_loss + g_loss)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(d_real[:, :-1], 1), tf.argmax(self.label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss, accuracy
        # }}}

        # G takes ramdon noise and tries to generate images [B, h, w, c]
        # G由四个反向卷积层组成,各层参数如下：前三层每层激活函数为relu，衰减率为0.9的BN；
        # 第一层卷积核为[2,2]，步长为[1,1]；跌第二层卷积核为[4,4]，步长为[2,2];第三层卷积核为[4,4]，步长为[2,2]
        # 第一层（64,2,2,384），第二层（64,6,6,128），第三层（64,14,14,64），第四层（64,32,32,3）
        def G(z, scope='Generator'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                z = tf.reshape(z, [self.batch_size, 1, 1, -1])
                g_1 = deconv2d(z, deconv_info[0], is_train, name='g_1_deconv')
                log.info('{} {}'.format(scope.name, g_1))
                g_2 = deconv2d(g_1, deconv_info[1], is_train, name='g_2_deconv')
                log.info('{} {}'.format(scope.name, g_2))
                g_3 = deconv2d(g_2, deconv_info[2], is_train, name='g_3_deconv')
                log.info('{} {}'.format(scope.name, g_3))
                g_4 = deconv2d(g_3, deconv_info[3], is_train, name='g_4_deconv', activation_fn=tf.tanh)
                log.info('{} {}'.format(scope.name, g_4))
                output = g_4
                assert output.get_shape().as_list() == self.image.get_shape().as_list(), output.get_shape().as_list()
            return output

        # D takes images as input and tries to output class label [B, n+1]
        # D由3个卷积层和1个全连接层组成，各层参数如下：每层卷积核为[5,5]，步长为[2,2]；
        # 激活函数为0.2倍的leaky relu，使用衰减率为0.9的BN，加上0.5的dropout；
        # 第一层输出（64,16,16,64），第二层输出（64,8,8,128），第三层输出（64,4,4,256），第四层输出（64,11）
        def D(img, scope='Discriminator', reuse=True):
            with tf.variable_scope(scope) as scope:
                filsize = (5, 5)
                inputShape = (32, 32, 3)
                channelAxis = -1
                time_routing = 3
                init_act = 'relu'
                bnArgs = {
                    "axis": channelAxis,
                    "momentum": 0.9,
                    "epsilon": 1e-04
                }
                convArgs = {
                    "padding": "same",
                    "use_bias": False,
                    "strides": (2, 2),
                    # "kernel_regularizer": KR.l2(0.0001),
                    # 是否在频域初始化
                    "spectral_parametrization": False,
                    "kernel_initializer": "ComplexIndependent"
                }

                # I = KL.Input(shape=inputShape)
                I = img
                if not reuse:
                    log.warn(scope.name)
                # 对实数输入通过网络处理生成对应的虚部
                input_img = learnConcatRealImagBlock(I, (1, 1), (3, 3), 0, '0', bnArgs, init_act)
                input_cpx = KL.Concatenate(channelAxis)([I, input_img])
                # 第一层复数卷积，复数BatchNorm和复数dropout
                conv_1 = ComplexConv2D(conv_info[0], filsize, name='d_1_conv', **convArgs)(input_cpx)
                bn_1 = complexBN(name='stage_1_bn', **bnArgs)(conv_1)
                drop_1 = KL.Dropout(rate=0.5)(bn_1)
                if not reuse:
                    log.info('{} {}'.format(scope.name, drop_1))
                # 第二层
                conv_2 = ComplexConv2D(conv_info[1], filsize, name='d_2_conv', **convArgs)(drop_1)
                bn_2 = complexBN(name='stage_1_bn', **bnArgs)(conv_2)
                drop_2 = KL.Dropout(rate=0.5)(bn_2)
                if not reuse:
                    log.info('{} {}'.format(scope.name, drop_2))
                # 第三层

                # conv_3 = ComplexConv2D(conv_info[2], filsize, name='d_3_conv', **convArgs)(drop_2)
                # bn_3 = complexBN(name='stage_1_bn', **bnArgs)(conv_3)
                # drop_3 = KL.Dropout(rate=0.5)(bn_3)
                # if not reuse:
                #     log.info('{} {}'.format(scope.name, drop_3))
                # # 全连接层
                # flat_3 = flatten_fully(drop_3)
                # d_4 = ComplexDense(int(n+1), name='fc', activation=None)(flat_3)
                # if not reuse:
                #     log.info('{} {}'.format(scope.name, d_4))
                # output = d_4

                caps_3 = PrimaryCap(drop_2, 8, 16, filsize, **convArgs)
                if not reuse:
                    log.info('{} {}'.format(scope.name, caps_3))
                caps_fc = CapsuleLayer(num_capsule=n+1, dim_capsule=16, routings=time_routing, name='caps_fc')(caps_3)
                output = shape_loss(caps_fc)
                if not reuse:
                    log.info('{} {}'.format(scope.name, output))
                assert output.get_shape().as_list() == [self.batch_size, n+1]
                return tf.nn.softmax(output), output

        # Generator {{{
        # =========
        # 输入随机噪声,生成假图片
        z = tf.random_uniform([self.batch_size, n_z], minval=-1, maxval=1, dtype=tf.float32)
        fake_image = G(z)
        self.fake_img = fake_image
        # }}}
        # 分别对真实和虚假图像进行判别，得到判别结果
        # Discriminator {{{
        # =========
        d_real, d_real_logits = D(self.image, scope='Discriminator', reuse=False)
        d_fake, d_fake_logits = D(fake_image, scope='Discriminator', reuse=True)
        self.all_preds = d_real
        self.all_targets = self.label
        # }}}

        self.S_loss, d_loss_real, d_loss_fake, self.d_loss, self.g_loss, GAN_loss, self.accuracy = \
            build_loss(d_real, d_real_logits, d_fake, d_fake_logits, self.label, self.image, fake_image)

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/GAN_loss", GAN_loss)
        tf.summary.scalar("loss/S_loss", self.S_loss)
        tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
        tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
        tf.summary.scalar("loss/d_loss_fake", tf.reduce_mean(d_loss_fake))
        tf.summary.scalar("loss/g_loss", tf.reduce_mean(self.g_loss))
        tf.summary.image("img/fake", fake_image)
        tf.summary.image("img/real", self.image, max_outputs=1)
        tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
        # tf.summary.image("label/pred_real", tf.reshape(d_real, [3, self.batch_size, n+1, 1]))
        # tf.summary.image("label/pred_fake", tf.reshape(d_fake, [1, self.batch_size, n+1, 1]))
        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
