import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import datetime
import os
from sgld_sampler import sgld_sample, sgld_sample_diag
from util import RunningStats
from pathlib import Path


class JEM:
    def __init__(self, logits, optimizer, replay_buffer_size=10000):
        """
        Create JEM object
        :param logits: Classifier logits to use for JEM
        :param optimizer: Optimizer to use
        :param replay_buffer_size: Size of replay buffer, default 10000
        """
        self.shape = logits.layers[0].input_shape[0][1:]
        self.replay_buffer = np.random.uniform(0, 1, (replay_buffer_size,) + self.shape)
        self.model_save_name = None
        self.metric_results_samples = None
        self.cb_results_energy = None
        self.best_values_metrics = None
        self.logits = logits
        self.optimizer = optimizer

        # Create energy function
        @tf.function
        def energ_fun(X, training=True):
            logits_ = self.logits(X, training=training)
            return -tf.math.reduce_logsumexp(logits_, axis=-1)

        self.energy = energ_fun

        # Create classifier
        @tf.function
        def classifier(X, training=True):
            logits_ = self.logits(X, training=training)
            return tf.nn.softmax(logits_)

        self.classifier = classifier

    def sample_sgld(self, x_init, num_steps_markov=tf.constant(25),
                    step_size=tf.constant(10.0), var=tf.constant(0.005), clip_thresh=tf.constant(0.01),
                    constrain_results=False):
        """
        Sample from the resulting model, using SGLD
        :param x_init: Initial SGLD state
        :param num_steps_markov: Number of MCMC transitions
        :param step_size: MCMC step size
        :param var: Variance of noise used during sampling
        :param clip_thresh: Gradient clipping threshold
        :param constrain_results: Whether to clip the results to the range [0, 1]
        :return: SGLD samples
        """
        return sgld_sample(self.energy, x_init, num_steps_markov, step_size,
                           var, clip_thresh=clip_thresh, constrain_results=constrain_results)

    def sample_replay_buffer(self, batch_size, uniform_bounds_lower, uniform_bounds_upper,
                             num_steps_markov=tf.constant(25), step_size=tf.constant(10.0),
                             var=tf.constant(0.005), clip_thresh=tf.constant(0.01), uniform_chance=0.05,
                             constrain_results=False):
        """
        Sample using the replay buffer. Some of the images are initialized using the replay buffer,
        some from noise. Can be configured using uniform_chance
        :param batch_size: How many samples to produce
        :param uniform_bounds_lower: Lower bounds of noise
        :param uniform_bounds_upper: Upper bounds of noise
        :param num_steps_markov: Number of MCMC transition steps to use
        :param step_size: MCMC step size to use
        :param var: Variance of noise used during sampling
        :param clip_thresh: Gradient clipping threshold
        :param uniform_chance: How many % of the initial samples to take from noise, rather than the replay buffer
        :param constrain_results: Whether to clip the results to the range [0, 1]
        :return: SGLD samples, Gradient magnitude of SGLD during sampling
        """
        num_uniform = np.int(np.ceil(batch_size * uniform_chance))
        num_buffer = np.int(np.floor(batch_size * (1.0 - uniform_chance)))

        uniform = tf.random.uniform((num_uniform,) + self.shape, minval=uniform_bounds_lower,
                                    maxval=uniform_bounds_upper)
        indices_replay_buffer = np.random.choice(np.arange(self.replay_buffer.shape[0]), num_buffer)
        replay = tf.convert_to_tensor(self.replay_buffer[indices_replay_buffer], dtype=tf.float32)
        initial_points = tf.concat((uniform, replay), axis=0)

        return sgld_sample_diag(self.energy, initial_points, num_steps_markov, step_size,
                                var, clip_thresh=clip_thresh, constrain_results=constrain_results)

    def _insert_into_replay_buffer(self, data, batch_size):
        indices_replay_buffer = np.random.choice(np.arange(self.replay_buffer.shape[0]), batch_size)
        self.replay_buffer[indices_replay_buffer] = data

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
        """
        Save model
        :param name: Name of the model save location
        :return: None
        """
        name_path = Path(name)
        name_path.mkdir(parents=True, exist_ok=True)
        orig = os.getcwd()
        os.chdir(name_path)
        self.logits.save_weights('model')
        np.savez('rpbuf.npz', self.replay_buffer)
        np.save('weights.npy', self.optimizer.get_weights())
        os.chdir(orig)

    def load_model(self, name):
        """
        Load model
        :param name: Name of the model save location
        :return: None
        """
        name_path = Path(name)
        orig = os.getcwd()
        os.chdir(name_path)
        self.rpbuf_handle = np.load('rpbuf.npz')
        opt_weights = np.load('weights.npy', allow_pickle=True)
        grad_vars = self.energy.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
        self.optimizer.set_weights(opt_weights)
        self.logits.load_weights('model')
        os.chdir(orig)
        self.replay_buffer = self.rpbuf_handle.f.arr_0

    def fit(self, X_train, y_train, batch_size, num_epochs, optimizer, uniform_bounds_lower, uniform_bounds_upper,
            alpha=tf.constant(0.1), num_steps_markov=tf.constant(25), step_size=tf.constant(1.0),
            var=tf.constant(0.005), clip_thresh=tf.constant(0.01), callbacks_energy=None,
            metrics_samples=None, save_best_weights=False, early_stopping=False, uniform_chance=0.05,
            weight_ce_loss=tf.constant(1.0), use_replay_buffer=True, injected_noise=tf.constant(1e-2),
            constrain_results=False):
        """
        Fit the JEM
        :param data: Data to use for fitting; should be numpy array
        :param batch_size: Batch size to use during fitting
        :param num_epochs: Number of training epochs
        :param uniform_bounds_lower: Lower bounds of the data domain
        :param uniform_bounds_upper: Upper bounds of the data domain
        :param alpha: L2 regularization magnitude for the energies
        :param num_steps_markov: Number of MCMC transition steps
        :param step_size: MCMC step size
        :param var: Variance of noise used during sampling
        :param clip_thresh: Gradient clipping threshold for SGLD samples
        :param callbacks_energy: Energy callbacks for every epoch
        :param metrics_samples: Metrics for the samples, like IS/FID
        :param save_best_weights: Whether to save the best weights so far
        :param early_stopping: Whether to stop if no improvements
        :param uniform_chance: Change of uniform noise for replay buffer
        :param constrain_results: Whether to constrain samples to the [0, 1] range
        :param injected_noise: How much noise to inject ot the training samples. Helps with stability
        :param use_replay_buffer: Whether to use the replay buffer for sampling
        :param weight_ce_loss: Weighing of the CE loss
        :return: None
        """

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
        n_train = X_train.shape[0]
        inner_loop_iterations = n_train // batch_size

        # Create dataset sampler
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=n_train)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset_iterator = iter(dataset)

        # Execute callbacks before training start
        self._handle_energy_callbacks(callbacks_energy)

        # Set name for saving the weights of this training run
        self.model_save_name = "ebm_{date:%Y-%m-%d_%H#%M#%S}".format(date=datetime.datetime.now())
        if save_best_weights:
            self.save_model(self.model_save_name)

        # Prepare to collect running statistics for metrics
        metric_vals_epoch = {}
        metric_vals_best_epoch = {}

        for metric_name, _ in metrics_samples:
            metric_vals_best_epoch[metric_name] = RunningStats()
            metric_vals_best_epoch[metric_name].push(float("-inf"))

        # Define CE loss
        scce = tf.keras.losses.SparseCategoricalCrossentropy()

        # Execute main training loop
        for epoch in range(num_epochs):
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
                sample_data_dist, labels = next(dataset_iterator)
                # Sample from data distribution
                if injected_noise is not None:
                    # Inject Gaussian noise
                    sample_data_dist = tf.cast(sample_data_dist, tf.dtypes.float32) + \
                                       tf.random.normal(tf.shape(sample_data_dist),
                                                        stddev=injected_noise,
                                                        dtype=tf.dtypes.float32)
                # Sample from energy function
                if use_replay_buffer:
                    sample_energy_dist, r_st = self.sample_replay_buffer(batch_size,
                                                                         uniform_bounds_lower, uniform_bounds_upper,
                                                                         num_steps_markov=num_steps_markov,
                                                                         step_size=used_step_size,
                                                                         var=used_var,
                                                                         clip_thresh=clip_thresh,
                                                                         uniform_chance=uniform_chance,
                                                                         constrain_results=constrain_results)
                    # Insert new samples into the replay buffer
                    self._insert_into_replay_buffer(sample_energy_dist, batch_size)
                else:
                    initial_points = tf.random.uniform(sample_data_dist.shape, minval=uniform_bounds_lower,
                                                       maxval=uniform_bounds_upper)
                    sample_energy_dist, r_st = sgld_sample_diag(self.energy, initial_points, num_steps_markov,
                                                                step_size,
                                                                var, clip_thresh=clip_thresh)

                # Compute weight gradients
                with tf.GradientTape() as g:
                    # Energy loss
                    energies_data = self.energy(sample_data_dist)
                    energies_samples = self.energy(sample_energy_dist)
                    energy_data = tf.math.reduce_mean(energies_data)
                    energy_samples = tf.math.reduce_mean(energies_samples)
                    if alpha > 0.0:
                        energies_l2 = tf.math.reduce_mean(tf.square(energies_data)) + \
                                      tf.math.reduce_mean(tf.square(energies_samples))
                        energy = energy_data - energy_samples + alpha * energies_l2
                    else:
                        energy = energy_data - energy_samples
                    # CE loss
                    pred = self.classifier(sample_data_dist)
                    ce_loss = scce(labels, pred)

                    loss = energy + weight_ce_loss * ce_loss

                gradient = g.gradient(loss, self.logits.trainable_variables)
                # Apply gradients
                optimizer.apply_gradients(zip(gradient, self.logits.trainable_variables))

                # Print epoch progress
                num_samples_processed += batch_size
                progress = num_samples_processed / n_train
                print(
                    "\rEpoch {} progress: {:06.2f}%, Energy data {:06.4f} samples {:06.4f} gradient magnitude {:06.4f}"
                    " CE loss {:06.4f}"
                    .format(epoch, progress * 100.0, energy_data, energy_samples, r_st, energy, scce), end='')

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
            score = self._score_metrics(metric_vals_epoch, metric_vals_best_epoch, metrics_samples)

            # If saving best weights is enabled, check if the metrics are better than average
            # and if yes, save the model to disk
            if epoch == 0 or score > 0:
                for metric_name, _ in metrics_samples:
                    self.best_values_metrics[metric_name] = metric_vals_epoch[metric_name].mean()
                    # Record metric statistics from best epoch
                    metric_vals_best_epoch = metric_vals_epoch
            self.save_model(self.model_save_name)
            metric_vals_epoch = {}

            # If early stopping is enabled, stop now if performance is below average
            if epoch > 0 and early_stopping and score <= 0.0:
                break

        return self.metric_results_samples
