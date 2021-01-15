import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import datetime
import os
from sgld_sampler import sgld_sample
from util import RunningStats
from pathlib import Path


class EBM:
    def __init__(self, shape, energy_fn, replay_buffer_size=10000):
        self.energy = energy_fn
        self.replay_buffer = np.random.uniform(0, 1, (replay_buffer_size,) + shape)
        self.weight_save_name = None
        self.metric_results_samples = None
        self.cb_results_energy = None
        self.best_values_metrics = None
        self.rpbuf_handle = None

    def sample_sgld(self, x_init, num_steps_markov=tf.constant(25),
                    step_size=tf.constant(10.0), var=tf.constant(0.005), clip_thresh=tf.constant(0.01)):
        return sgld_sample(self.energy, x_init, num_steps_markov, step_size,
                           var, clip_thresh=clip_thresh)

    def sample_replay_buffer(self, shape, batch_size, uniform_bounds_lower, uniform_bounds_upper,
                             num_steps_markov=tf.constant(25), step_size=tf.constant(10.0),
                             var=tf.constant(0.005), clip_thresh=tf.constant(0.01), uniform_chance=0.05):
        num_uniform = np.int(np.ceil(batch_size * uniform_chance))
        num_buffer = np.int(np.floor(batch_size * (1.0 - uniform_chance)))

        uniform = tf.random.uniform((num_uniform,) + shape, minval=uniform_bounds_lower, maxval=uniform_bounds_upper)
        indices_replay_buffer = np.random.choice(np.arange(self.replay_buffer.shape[0]), num_buffer)
        replay = tf.convert_to_tensor(self.replay_buffer[indices_replay_buffer], dtype=tf.float32)
        initial_points = tf.concat((uniform, replay), axis=0)

        return sgld_sample(self.energy, initial_points, num_steps_markov, step_size,
                           var, clip_thresh=clip_thresh)

    def _insert_into_replay_buffer(self, data, batch_size):
        indices_replay_buffer = np.random.choice(np.arange(self.replay_buffer.shape[0]), batch_size)
        self.replay_buffer[indices_replay_buffer] = data

    def sample_using_replay_buffer(self, num_steps, shape, batch_size, num_steps_markov=100, step_size=10, var=0.005 ** 2,
                                   clip_thresh=0.01, uniform_chance=0.05):
        # Sample from energy function
        for _ in range(num_steps):
            sample_energy_dist = self.sample_replay_buffer(shape, batch_size,
                                                           0, 1,
                                                           num_steps_markov=num_steps_markov,
                                                           step_size=step_size,
                                                           var=var,
                                                           clip_thresh=clip_thresh,
                                                           uniform_chance=uniform_chance)
            # Insert new samples into the replay buffer
            self._insert_into_replay_buffer(sample_energy_dist, batch_size)
        return sample_energy_dist

    def _handle_energy_callbacks(self, callbacks_energy):
        for callbacks_name, callback_fn in callbacks_energy:
            value_cb = callback_fn(self.energy)
            if callbacks_name in self.cb_results_energy:
                self.cb_results_energy[callbacks_name].append(value_cb)
            else:
                self.cb_results_energy[callbacks_name] = []
                self.cb_results_energy[callbacks_name].append(value_cb)

    @staticmethod
    def _handle_metrics(metric_results_samples, metric_vals_epoch, metrics_samples, sample_data_dist,
                        sample_energy_dist, it):
        for metric_name, callback_fn in metrics_samples:
            value_metric = callback_fn(sample_data_dist, sample_energy_dist, it)
            if metric_name in metric_results_samples:
                metric_results_samples[metric_name].append(value_metric)
            else:
                metric_results_samples[metric_name] = []
                metric_results_samples[metric_name].append(value_metric)
            metric_vals_epoch[metric_name].push(value_metric)
            print(" ", metric_name, "{:06.4f}".format(metric_vals_epoch[metric_name].mean()), end='')

    @staticmethod
    def _score_metrics(metric_vals_epoch, metric_vals_last_epoch, metrics_samples):
        score = 0.0
        for metric_name, _ in metrics_samples:
            avg_current = metric_vals_epoch[metric_name].mean()
            avg_last = metric_vals_last_epoch[metric_name].mean()
            if avg_current > avg_last:
                score += 1.0
            else:
                score -= 1.0
        return score

    def save_model(self, name):
        name_path = Path(name)
        name_path.mkdir(parents=True, exist_ok=True)
        orig = os.getcwd()
        os.chdir(name_path)
        self.energy.save_weights('model')
        np.savez('rpbuf.npz', self.replay_buffer)
        os.chdir(orig)

    def load_model(self, name):
        name_path = Path(name)
        orig = os.getcwd()
        os.chdir(name_path)
        self.energy.load_weights('model')
        self.rpbuf_handle = np.load('rpbuf.npz')
        os.chdir(orig)
        self.replay_buffer = self.rpbuf_handle.f.arr_0

    def fit(self, data, batch_size, num_epochs, optimizer, uniform_bounds_lower, uniform_bounds_upper,
            alpha=tf.constant(0.1), num_steps_markov=tf.constant(25), step_size=tf.constant(10.0),
            var=tf.constant(0.005), clip_thresh=tf.constant(0.01), callbacks_energy=None,
            metrics_samples=None, save_best_weights=False, early_stopping=False, uniform_chance=0.05):
        # Initialize metrics and callbacks
        if metrics_samples is None:
            metrics_samples = []
        if callbacks_energy is None:
            callbacks_energy = []
        # This is a map, mapping metric name -> list of values.
        # A value is recorded for each metric each batch
        self.metric_results_samples = {}
        # This is for results of callbacks on the energy function
        self.cb_results_energy = {}
        # This tracks the best, epoch-averaged values of the metrics. Best meaning the
        # epoch that had the highest average values of these metrics.
        # Maps metric name -> best value, epoch-average
        self.best_values_metrics = {}

        # Get number of training examples
        n_train = data.shape[0]
        inner_loop_iterations = n_train // batch_size

        # Create dataset sampler
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=n_train)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset_iterator = iter(dataset)

        # Execute callbacks before training start
        self._handle_energy_callbacks(callbacks_energy)

        # Set name for saving the weights of this training run
        self.weight_save_name = "ebm_{date:%Y-%m-%d_%H#%M#%S}".format(date=datetime.datetime.now())
        if save_best_weights:
            self.energy.save_weights(self.weight_save_name)

        # Prepare to collect running statistics for metrics
        metric_vals_epoch = {}
        metric_vals_last_epoch = {}

        for metric_name, _ in metrics_samples:
            metric_vals_last_epoch[metric_name] = RunningStats()
            metric_vals_last_epoch[metric_name].push(float("-inf"))

        # Execute main training loop
        for epoch in range(num_epochs):
            print("Epoch {}: 000.00%".format(epoch))
            num_samples_processed = 0

            # Execute schedules for noise standard deviation/step size
            used_var = var
            if callable(var):
                used_var = var(epoch)
            used_step_size = step_size
            if callable(step_size):
                used_step_size = step_size(epoch)

            # Reset epoch-wide averages of metrics
            for metric_name, _ in metrics_samples:
                metric_vals_epoch[metric_name] = RunningStats()

            # Go over training examples in epoch
            for i in range(inner_loop_iterations):
                # Sample from data distribution
                sample_data_dist = next(dataset_iterator)
                # Sample from energy function
                sample_energy_dist = self.sample_replay_buffer(sample_data_dist.shape[1:], batch_size,
                                                               uniform_bounds_lower, uniform_bounds_upper,
                                                               num_steps_markov=num_steps_markov,
                                                               step_size=used_step_size,
                                                               var=used_var,
                                                               clip_thresh=clip_thresh,
                                                               uniform_chance=uniform_chance)
                # Insert new samples into the replay buffer
                self._insert_into_replay_buffer(sample_energy_dist, batch_size)

                # Compute weight gradients
                with tf.GradientTape() as g:
                    energies_data = self.energy(sample_data_dist, training=True)
                    energies_samples = self.energy(sample_energy_dist, training=True)
                    energy_data = tf.math.reduce_mean(energies_data)
                    energy_samples = tf.math.reduce_mean(energies_samples)
                    energies_l2 = tf.math.reduce_mean(tf.square(energies_data)) + \
                                  tf.math.reduce_mean(tf.square(energies_samples))
                    energy = energy_data - energy_samples + alpha * energies_l2
                gradient = g.gradient(energy, self.energy.trainable_variables)

                # Apply gradients
                optimizer.apply_gradients(zip(gradient, self.energy.trainable_variables))

                # Print epoch progress
                num_samples_processed += batch_size
                progress = num_samples_processed / n_train
                print(
                    "\rEpoch progress: {:06.2f}%, Energy data {:06.4f} samples {:06.4f} overall {:06.4f}".format(
                        progress * 100.0, energy_data, energy_samples, energy), end='')
                # Execute metric callbacks, evaluate metrics
                self._handle_metrics(self.metric_results_samples, metric_vals_epoch, metrics_samples,
                                     sample_data_dist, sample_energy_dist, i)

            # Execute energy function callbacks after every epoch
            self._handle_energy_callbacks(callbacks_energy)
            print()

            # Extract epoch-averaged values for metrics; compare them to the
            # best values so far
            # score > 0 => improvement, score < 0 => got worse,
            # score == 0 => ambiguous
            # Ties (score == 0) are broken by assuming worse
            score = self._score_metrics(metric_vals_epoch, metric_vals_last_epoch, metrics_samples)

            # If saving best weights is enabled, check if the metrics are better than average
            # and if yes, save the model to disk
            if score > 0:
                for metric_name, _ in metrics_samples:
                    self.best_values_metrics[metric_name] = metric_vals_epoch[metric_name].mean()
            if save_best_weights and score > 0.0:
                self.energy.save_weights(self.weight_save_name)

            # Record metric statistics from last epoch, reset current statistics
            metric_vals_last_epoch = metric_vals_epoch
            metric_vals_epoch = {}

            # If early stopping is enabled, stop now if performance is below average
            if epoch > 0 and early_stopping and score <= 0.0:
                break

        return self.metric_results_samples
