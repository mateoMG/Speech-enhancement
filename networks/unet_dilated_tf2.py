import numpy as np
import tensorflow as tf

class StridedConv(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(StridedConv, self).__init__()
        self.n_filters = n_filters

        self.conv1 = tf.keras.layers.Conv2D(48, (3, 3), strides=(2, 1), padding='SAME', activation=None)
    def call(self, x):
        return self.conv1(x)

class StridedDeconv(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(StridedDeconv, self).__init__()
        self.n_filters = n_filters

        self.conv1 = tf.keras.layers.Conv2DTranspose(2, (6, 3), strides=(2, 1), padding='SAME', activation=None)
    def call(self, x):
        return self.conv1(x)


class BNELU(tf.keras.layers.Layer):
    def __init__(self):
        super(BNELU, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-4)
        self.elu = tf.keras.layers.ELU()
    def call(self, x, training=False):
        x = self.bn(x, training=training)
        return self.elu(x)

class RTBlock(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(RTBlock, self).__init__()
        self.conv_net1 = tf.keras.layers.Conv2D(
                num_units/2, (1, 9), dilation_rate=(1, 4), padding='SAME', activation=None)
        self.bnelu_net1 = BNELU()
        self.conv_net2 = tf.keras.layers.Conv2D(
                num_units/2, (1, 9), dilation_rate=(1, 16), padding='SAME', activation=None)
        self.bnelu_net2 = BNELU()
    def call(self, x, training=None):
        net1 = self.conv_net1(x)
        net1 = self.bnelu_net1(net1, training=training)
        net2 = self.conv_net2(x)
        net2 = self.bnelu_net2(net2, training=training)
        return tf.concat([x, net1, net2], axis=-1)

class RFBlock(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(RFBlock, self).__init__()
        self.conv_net1 = tf.keras.layers.Conv2D(
                num_units/2, (9, 1), dilation_rate=(3, 1), padding='SAME', activation=None)
        self.bnelu_net1 = BNELU()
        self.conv_net2 = tf.keras.layers.Conv2D(
                num_units/2, (9, 1), dilation_rate=(9, 1), padding='SAME', activation=None)
        self.bnelu_net2 = BNELU()
    def call(self, x, training=None):
        net1 = self.conv_net1(x)
        net1 = self.bnelu_net1(net1, training=training)
        net2 = self.conv_net2(x)
        net2 = self.bnelu_net2(net2, training=training)
        return tf.concat([x, net1, net2], axis=-1)


class WCU_CT(tf.keras.layers.Layer):
    def __init__(self, K, H_ctx):
        super(WCU_CT, self).__init__()
        self.RT = RTBlock(H_ctx)
        self.conv = tf.keras.layers.Conv2D(
                K, (3, 3), padding='SAME', activation=None)
    def call(self, x, training=None):
        x = self.RT(x, training=training)
        x = self.conv(x)
        return x

class WCU_CF(tf.keras.layers.Layer):
    def __init__(self, K, H_ctx):
        super(WCU_CF, self).__init__()
        self.RF = RFBlock(H_ctx)
        self.conv = tf.keras.layers.Conv2D(
                K, (3, 3), padding='SAME', activation=None)
    def call(self, x, training=None):
        x = self.RF(x, training=training)
        x = self.conv(x)
        return x

class WCU_DT(tf.keras.layers.Layer):
    def __init__(self, K, H_ctx):
        super(WCU_DT, self).__init__()
        self.RT = RTBlock(H_ctx)
        self.conv = tf.keras.layers.Conv2DTranspose(
                K, (3, 3), padding='SAME', activation=None)
    def call(self, x, training=None):
        x = self.RT(x, training=training)
        x = self.conv(x)
        return x

class WCU_DF(tf.keras.layers.Layer):
    def __init__(self, K, H_ctx):
        super(WCU_DF, self).__init__()
        self.RF = RFBlock(H_ctx)
        self.conv = tf.keras.layers.Conv2DTranspose(
                K, (3, 3), padding='SAME', activation=None)
    def call(self, x, training=None):
        x = self.RF(x, training=training)
        x = self.conv(x)
        return x





class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.strided_conv = StridedConv(48)
        self.bnelus =  [BNELU() for i in range(9)]
        self.WCU_1C = WCU_CT(48, 16)
        self.WCU_2C = WCU_CF(48, 16)
        self.WCU_3C = WCU_CT(48, 16)
        self.WCU_4C = WCU_CF(48, 16)
        self.WCU_4D = WCU_DT(48, 16)
        self.WCU_3D = WCU_DF(48, 16)
        self.WCU_2D = WCU_DT(48, 16)
        self.WCU_1D = WCU_DF(48, 16)
        self.strided_deconv = StridedDeconv(2)

    @staticmethod
    def train_length():
        return 500

    @staticmethod
    def num_freq():
        return 256

    @staticmethod
    def learning_rate():
        return 0.01

    def call(self, x, training=False):
        n_frames = tf.shape(x)[3]
        net = tf.transpose(x, (0, 2, 3, 1))
        net = self.strided_conv(net)
        net_lvl1 = net
        net = self.bnelus[0](net, training=training)
        net = self.WCU_1C(net, training=training)
        net_lvl2 = net
        net = self.bnelus[1](net, training=training)
        net = self.WCU_2C(net, training=training)
        net_lvl3 = net
        net = self.bnelus[2](net, training=training)
        net = self.WCU_3C(net, training=training)
        net_lvl4 = net
        net = self.bnelus[3](net, training=training)
        net = self.WCU_4C(net, training=training)
        net = self.bnelus[4](net, training=training)
        net = self.WCU_4D(net, training=training)
        net = net + net_lvl4
        net = self.bnelus[5](net, training=training)
        net = self.WCU_3D(net, training=training)
        net = net + net_lvl3
        net = self.bnelus[6](net, training=training)
        net = self.WCU_2D(net, training=training)
        net = net + net_lvl2
        net = self.bnelus[7](net, training=training)
        net = self.WCU_1D(net, training=training)
        net = net + net_lvl1
        net = self.bnelus[8](net, training=training)
        net = self.strided_deconv(net)
        net = tf.transpose(net, (0, 3, 1, 2))
        return net






