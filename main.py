#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import skimage
from sgld_sampler import sgld_sample
from EBM import EBM
from wrn import create_wide_residual_network
from wrn_spec import create_wide_residual_network_spectral_norm
from scores import inception_score, frechet_inception_distance


def fuse_images(width, height, images, img_width, img_height, n_channels=3):
    really_big_image = None
    for i in range(width):
        big_image = None
        for j in range(height):
            cur_image = images[width * i + j].reshape(img_width, img_height, n_channels)
            if big_image is not None:
                big_image = np.hstack((big_image, cur_image))
            else:
                big_image = cur_image
        if really_big_image is not None:
            really_big_image = np.vstack((really_big_image, big_image))
        else:
            really_big_image = big_image
    return really_big_image


def main():
    # Load CIFAR-10 dataset
    (X_train_full, y_train_full), (_, _) = keras.datasets.cifar10.load_data()
    X_train_full = X_train_full / 255.0

    # Create classifier for IS/FID
    classifier_model = create_wide_residual_network((32, 32, 3), nb_classes=10, N=4, k=8)
    classifier_model.load_weights('class-1')

    # Callbacks for IS/FID
    def is_metric_callback(samples_data, samples_energ, it):
        p_yx = tf.nn.softmax(classifier_model(samples_energ)).numpy()
        score = inception_score(p_yx)
        if it == 0:
            imgs = np.array(samples_energ)
            imgs = np.clip(imgs, 0, 1)
            img = fuse_images(10, 10, imgs, 32, 32)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.colorbar()
            plt.show()
        return score

    # Need last-layer activations for FID
    cifar_statistics = keras.Model(classifier_model.input, classifier_model.get_layer(name='flatten').output)

    def fid_metric_callback(samples_data, samples_energ, it):
        stats_gen = cifar_statistics(samples_energ)
        stats_real = cifar_statistics(samples_data)
        score = frechet_inception_distance(stats_real, stats_gen)
        return -score

    # Create model for energy function
    ebm_model = create_wide_residual_network((32, 32, 3), nb_classes=1, N=4, k=8)

    ebm = EBM((32, 32, 3), ebm_model)
    lr = 1e-4
    batch_size = 128
    step_size = 15
    num_epochs = 25
    optimizer = keras.optimizers.Adam(lr)

    ebm.fit(X_train_full, batch_size, 5, optimizer, 0.0, 1.0, num_steps_markov=tf.constant(60),
            var=tf.constant(0.005 ** 2), step_size=step_size, callbacks_energy=[],
            metrics_samples=[("IS", is_metric_callback), ("FID", fid_metric_callback)],
            alpha=1.0, clip_thresh=tf.constant(0.01), save_best_weights=True, early_stopping=False,
            uniform_chance=0.05)
    ebm_model.save_weights("model_weights_5")
    ebm.fit(X_train_full, batch_size, num_epochs - 5, optimizer, 0.0, 1.0, num_steps_markov=tf.constant(60),
            var=tf.constant(0.005 ** 2), step_size=step_size, callbacks_energy=[],
            metrics_samples=[("IS", is_metric_callback), ("FID", fid_metric_callback)],
            alpha=1.0, clip_thresh=tf.constant(0.01), save_best_weights=True, early_stopping=False,
            uniform_chance=0.05)
    ebm.save_model("ebm-model")


if __name__ == "__main__":
    main()
