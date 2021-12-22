import tensorflow as tf


class Resnet(tf.keras.layers.Layer):
    def __init__(self, filters, strides, activation="relu"):
        super(Resnet, self).__init__()
        self.f = filters
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()]


    def call(self, inputs):
        X = inputs
        #print(self.f)
        for layer in self.main_layers:
            X = layer(X)
        skip_X = inputs
        for layer in self.skip_layers:
            skip_X = layer(skip_X)
        return self.activation(X + skip_X)



class MyNetwork(tf.keras.Model):
    def __init__(self, n_classes):
        super(MyNetwork, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(64, 7, strides=2, input_shape=[256, 1000, 2], padding='same', use_bias=False)
        self.drop = tf.keras.layers.Dropout(0.5)
        self.batch = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_1 = Resnet(64, 1)
        self.resnet_2 = Resnet(64, 1)
        self.resnet_3 = Resnet(64, 1)
        self.resnet_4 = Resnet(128, 2)
        self.resnet_5 = Resnet(128, 1)
        self.resnet_6 = Resnet(128, 1)
        self.resnet_7 = Resnet(128, 1)
        self.resnet_8 = Resnet(256, 2)
        self.resnet_9 = Resnet(256, 1)
        self.resnet_10 = Resnet(256, 1)
        self.resnet_11 = Resnet(256, 1)
        self.resnet_12 = Resnet(256, 1)
        self.resnet_13 = Resnet(256, 1)
        self.resnet_14 = Resnet(512, 2)
        self.resnet_15 = Resnet(512, 1)
        self.resnet_16 = Resnet(512, 1)

        self.globalavg = tf.keras.layers.GlobalAvgPool2D()
        self.flatten = tf.keras.layers.Flatten()

        self.dense_outpur = tf.keras.layers.Dense(n_classes, activation="softmax")


    def call(self, inputs):
        network = tf.transpose(inputs, (0, 2, 3, 1))
        network = self.conv_1(network)
        network = self.drop(network)
        network = self.batch(network)
        network = self.relu(network)
        network = self.max_pool(network)

        network = self.resnet_1(network)
        network = self.resnet_2(network)
        network = self.resnet_3(network)
        network = self.resnet_4(network)
        network = self.resnet_5(network)
        network = self.resnet_6(network)
        network = self.resnet_7(network)
        network = self.resnet_8(network)
        network = self.resnet_9(network)
        network = self.resnet_10(network)
        network = self.resnet_11(network)
        network = self.resnet_12(network)
        network = self.resnet_13(network)
        network = self.resnet_14(network)
        network = self.resnet_15(network)
        network = self.resnet_16(network)
        network = self.globalavg(network)
        network = self.flatten(network)
        network = self.dense_outpur(network)

        return network





