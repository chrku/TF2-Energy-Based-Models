import tensorflow.keras as keras
import tensorflow_addons as tfa


def vgg(layer_in, n_filters, n_conv):
    """
    VGG block
    :param layer_in: input layer
    :param n_filters: number of filters to use
    :param n_conv: number of convolutional layers to use
    :return: VGG block
    """
    for _ in range(n_conv):
        layer_in = keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    layer_in = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(layer_in)
    return layer_in


def vgg_block_avg(layer_in, n_filters, n_conv):
    """
    VGG block with average pooling
    :param layer_in: input layer
    :param n_filters: number of filters to use
    :param n_conv: number of convolutional layers to use
    :return: VGG block
    """
    # Add convolutional layers
    for _ in range(n_conv):
        layer_in = keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    # Add max pooling layer
    layer_in = keras.layers.AveragePooling2D(pool_size=2)(layer_in)
    return layer_in


def inception_naive(layer_in, f1, f2, f3):
    """
    Naive inception module
    :param layer_in: input layer
    :param f1: number of 1x1 convolutions
    :param f2: number of 3x3 convolutions
    :param f3: number of 5x5 convolutions
    :return: inception module
    """
    conv1 = keras.layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = keras.layers.Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)
    conv5 = keras.layers.Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)
    pool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def inception(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    """
    Inception module
    :param layer_in: input layer
    :param f1: number of 1x1 convolutions
    :param f2: number of 3x3 convolutions
    :param f3: number of 5x5 convolutions
    :return: inception module
    """
    conv1 = keras.layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = keras.layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = keras.layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    conv5 = keras.layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = keras.layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    pool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = keras.layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters, use_spectral_norm=True):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = keras.layers.Conv2D(n_filters, (1, 1), padding='same', activation='swish',
                                          kernel_initializer='he_normal')(layer_in)
    if not use_spectral_norm:
        conv1 = keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='swish',
                                    kernel_initializer='he_normal', use_bias=False)(layer_in)
        conv2 = keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                                    kernel_initializer='he_normal', use_bias=False)(conv1)
    else:
        conv1 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='swish',
                                    kernel_initializer='he_normal', use_bias=False))(layer_in)
        conv2 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                                    kernel_initializer='he_normal', use_bias=False))(conv1)
    layer_out = keras.layers.add([conv2, merge_input])
    layer_out = keras.layers.Activation('swish')(layer_out)
    return layer_out
