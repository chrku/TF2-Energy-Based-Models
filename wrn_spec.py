from tensorflow.keras.layers import Convolution2D, AveragePooling2D
from tensorflow.keras.layers import Input, Add, Activation, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


def initial_conv(input, reg=None):
    x = tfa.layers.SpectralNormalization(Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(input)
    x = Activation('swish')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1), reg=None):
    x = tfa.layers.SpectralNormalization(Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(init)
    x = Activation('swish')(x)
    x = tfa.layers.SpectralNormalization(Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)
    skip = tfa.layers.SpectralNormalization(Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                         kernel_regularizer=reg,
                         use_bias=False))(init)
    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, reg=None):
    init = input
    x = Activation('swish')(input)
    x = tfa.layers.SpectralNormalization(Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)
    x = Activation('swish')(x)
    x = tfa.layers.SpectralNormalization(Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, reg=None):
    init = input
    x = Activation('swish')(input)
    x = tfa.layers.SpectralNormalization(Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)
    x = Activation('swish')(x)
    x = tfa.layers.SpectralNormalization(Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, reg=None):
    init = input
    x = Activation('swish')(input)
    x = tfa.layers.SpectralNormalization(Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)
    x = Activation('swish')(x)
    x = tfa.layers.SpectralNormalization(Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=reg,
                      use_bias=False))(x)
    m = Add()([init, x])
    return m


def create_wide_residual_network_spectral_norm(input_dim, nb_classes=100, N=2, k=1, reg=None):
    ip = Input(shape=input_dim)
    x = initial_conv(ip, reg=reg)
    nb_conv = 4
    x = expand_conv(x, 16, k, reg=reg)
    nb_conv += 2
    for i in range(N - 1):
        x = conv1_block(x, k, reg=reg)
        nb_conv += 2
    x = Activation('swish')(x)
    x = expand_conv(x, 32, k, strides=(2, 2), reg=reg)
    nb_conv += 2
    for i in range(N - 1):
        x = conv2_block(x, k, reg=reg)
        nb_conv += 2
    x = Activation('swish')(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_regularizer=reg)(x)

    model = Model(ip, x)

    return model
