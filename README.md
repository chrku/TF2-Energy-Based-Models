# TF2-Energy-Based-Models


Introduction
============

This main repository deals with implementing the techniques
presented in the papers of Du and Mordatch (2019) [@du2019implicit] and
Grathwohl et al. (2019) [@grathwohl2019your]. I implemented the main
techniques in these papers (using Tensorflow 2.0) and applied them to
various datasets.

How to use the code to train Energy-Based Models and the Joint-Energy Model can be seen in
the Jupyter notebooks included with the code.

Energy-based models
===================

To research energy-based models, I used a reference by Yann LeCun et al.
[@lecun2006tutorial]. EBMs are a general unsupervised learning technique
where the goal is to learn an energy function $E_{\theta}$, which maps
from a $d$-dimensional data space $\mathcal{X}$ to a real-valued energy
$E$. The goal is to shape the energy function in such a way that values
from the data manifold have a low energy and values outside the data
manifold have a high value. Such an energy function can be turned into a
probability density function by means of a Boltzmann-Gibbs distribution:

$$p(\bm{x}) = \frac{\exp(-E_{\theta}(\bm{x}))}{\int_\mathcal{X} \exp(-E_{\theta}(\bm{x})) d \bm{x} }$$

Evaluating the integral in the denominator (the partition function) is
in general intractable, which is why this is usually not done in
practice. EBMs are a rather old technique and well-known in both machine
learning and statistical physics.

Because of the intractability of the partition function, sampling-based
approaches are usually used in conjunction with EBMs. There exists a
rich body of literature for sampling from unnormalized probability
models like EBMs; very often, Markov Chain Monte Carlo (MCMC) methods
are used. This is a general class of methods that aims to sample from a
given distribution by constructing a Markov Chain which has the
distribution as its stationary distribution. If such a Markov Chain has
been constructed, sampling can proceed after evaluating the Markov Chain
for a given number of steps (the burn-in time).

The contribution by Du and Mordatch [@du2019implicit] was showing that
EBMs can be used alongside with deep neural networks to build a
generative model for images. The energy function $E_\theta$ is a deep
neural network, which takes in an image as an input and outputs a scalar
real-valued energy. $\theta$ are the weights of the neural network. No
particular constraints have to placed on its structure; this is one of
the advantages of the approach. Training proceeds by maximum likelihood;
i.e. we aim to maximize the likelihood of the data, which is equivalent
to minimizing the negative log-likelihood. Hence, we aim to minimize
$\mathcal{L}_\theta = \mathbb{E}_{\bm{x} \sim p_D}[-\log p_\theta(\bm{x})]$.
It can be shown [@du2019implicit] that the gradient of this objective
can be approximated with:

$$\nabla_\theta \mathcal{L} \approx \mathbb{E}_{\bm{x}^+ \sim p_D}[\nabla_\theta E_\theta(\bm{x}^+)] - 
        \mathbb{E}_{\bm{x}^- \sim q_\theta}[\nabla_\theta E_\theta(\bm{x}^-)]$$

where $q_\theta$ is the distribution defined by the energy function.
This distribution is approximated using MCMC. An intuitive explanation
of this process is that we depress the energy values of the data and
increase the energy values of the negative samples from the energy
function.

Of course, we will need a suitable way to generate samples from the
energy distribution. Du and Mordatch [@du2019implicit] propose using
Langevin Monte Carlo (LMC), which is a gradient-based MCMC method. LMC
includes a Metropolis-Hastings acceptance step; this is omitted and the
form used in the paper is:

$$\bm{x}^k = \bm{x}^{k - 1} - \frac{\lambda^2}{2} \nabla_{\bm{x}} E_{\theta}(\bm{x}^{k - 1}) + \bm{\omega}^k,
        \bm{\omega}^k \sim \mathcal{N}(0, \lambda \bm{I})$$

This will converge to the true distribution if $k \to \infty$ and
$\lambda \to 0$. In practice, a finite number of steps is used. This
process is quite computationally expensive, requiring one backward pass
per Langevin step. I will also note that it seems that very few papers
and implementations truly try to run convergent MCMC; mostly, short-run
chains that do not converge seem to be used, because of compute
constraints. Also, the noise variance often seems to be decoupled from
the step size. I also mostly used short-run MCMC.

Using deep neural networks as energy functions, training them with ML
and sampling using LMC is the main idea behind the paper of Du and
Mordatch [@du2019implicit]. They also use a replay buffer of past
samples to initialize the sampling process.

EBMs on toy data
----------------

I implemented the ideas referenced in the last section using Tensorflow
and Keras. To see if my implementation worked, I applied the idea on toy
distributions. These are 2-dimensional distributions of points which are
very easy to sample from and where the partition function is tractable
to calculate. To have a baseline to compare against, I also used
kernel-density estimation (KDE) on the distributions. I used a very
simple neural network architecture for all distributions; 2
fully-connected hidden layers with 1000 units each, using a ReLU
nonlinearity. I used 10 Langevin steps and a Langevin step size of 0.1
for all experiments. I found that the EBM was able to approximate the
distributions very well, though there sometimes were some outliers in
the sampling. I show the results of these experiments in
[8](#fig:toy-squares){reference-type="ref" reference="fig:toy-squares"},
[12](#fig:toy-circles){reference-type="ref" reference="fig:toy-circles"}
and [16](#fig:toy-spiral){reference-type="ref"
reference="fig:toy-spiral"}.

EBMs on MNIST
-------------

I chose the MNIST dataset for my first image dataset to evaluate the EBM
with. I used a rather simple convolutional neural network architecture. It
consists of 5 convolutional layers that gradually increases in the
filter size and have strides of 2. I used the Swish nonlinearity and
used no forms of regularization. I later found that image dataset
augmentation works well with EBMs; particularly adding a small amount of
Gaussian noise to the images, as that 'smooths' the data distribution. I
used a sample replay buffer, as was done in the paper. I trained the
network for 100 epochs.

A paper by Nijkamp et al. [@nijkamp2020anatomy] explores the training
process of EBMs. It was suggested to track the gradient magnitude in the
LMC sampling process to assess training stability; I found this to work
well.

To assess the visual quality of the reconstructions, I implemented the
inception score (IS) [@salimans2016improved] and the Fréchet Inception
Distance (FID) [@heusel2017gans]. The inception score (IS) is obtained
by training a classifier on the dataset and then calculating the
following statistic on the generated images:

$$S_{\mathrm{IS}} = \exp\left(\mathbb{E}_{\bm{x} \sim p_g} D_{\mathrm{KL}}(p(y | \bm{x}) || p(y))\right)$$

Here, $p(y | \bm{x})$ is the conditional distribution of the classifier
given the generated image $\bm{x}$ and $p(y)$ is the marginal
distribution, obtained as
$p(y) = \int_{\bm{x}} p(y | \bm{x}) p_g(\bm{x}) \approx \frac{1}{N} \sum_{\bm{x}} p(y | \bm{x})$.
The inception score attempts to measure both the diversity of generated
images and the peakedness of the conditional distribution. Higher values
are better.

The Fréchet Inception Distance (FID) compares the statistics that are
used by the classifier (e.g. the learned filters of the CNN, before the
last layer) using the Fréchet distance, which is a distance measure
defined for curves. Let $\mu_g$ be the mean of the statistics of a
collection of generated images and $\Sigma_g$ be the covariance of these
statistics and $\mu_r$ and $\Sigma_r$ are the mean and covariance of the
statistics of a collection of real images. Then the FID is defined as:

$$S_{\mathrm{FID}} = || \mu_r - \mu_g||^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1 / 2})$$

This attempts to measure the difference in statistics of generated vs.
real images. Lower values are better. I achieved a IS of about 7.67 and
a FID of about 89.09 on my model. Some generated image samples of the
model are shown in [1](#fig:mnist){reference-type="ref"
reference="fig:mnist"} The reconstructions aren't perfect, but I found
it hard to get good reconstructions using a reasonable amount of
compute. I probably could have trained for far more epochs; the IS and
FID were not done improving when I stopped.

![Samples for
MNIST[]{label="fig:mnist"}](imgs/ebm-samples-mnist.png){#fig:mnist
width="50%"}

### Notes on the training process

I noticed that using a replay buffer can make the training unstable; in
particular, the behaviour I noticed is that the model can, after a long
time (70-100 epochs) suddenly begin diverging, ruining the entire
training process. This does not seem to be a bug, as Grathwohl et al.
reported similar behaviour [@grathwohl2019your]. It seems to be caused
by 'bad' examples in the replay buffer, but I do not know the exact
mechanism. Unfortunately, I do not have a good solution to this problem,
other than usage of training checkpoints. Du and Mordatch seem to use
heavy amounts of gradient clipping and spectral normalization to combat
this problem; I found that using a lot of MCMC steps tends to help avoid
this problem, which comes at the cost of making the training process
more expensive. The replay buffer does speed up convergence
significantly, so not using one comes at a cost. Nijkamp et al.
[@nijkamp2020anatomy] reported being able to train networks without
using one, without any other significant adjustments, other than a heavy
increase in the amoung of time it takes to train the model.

EBMs on CIFAR-10
----------------

For CIFAR-10, I ended up using a very similar network to the one I used
for MNIST, with about 700000 parameters. This is notably smaller than
the size used in the paper of Du and Mordatch [@du2019implicit]; I tried
to chose the best size for my time and compute budget. I used 150 MCMC
steps, a Langevin step size of 1 and trained for 1000 epochs, which
amounted to 48 hours wall-clock time. The results can be seen in
[2](#fig:cifar){reference-type="ref" reference="fig:cifar"}. I achieved
an inception score of 3.8 and a FID of 140.7.

![Samples for CIFAR[]{label="fig:cifar"}](imgs/cifar-images){#fig:cifar
width="50%"}

I also did experiments in using the model for denoising, inpainting and
out-of-distribution detection. In these experiments, one can see that
the images get more saturated and that the model tends towards piecewise
constant solutions as the number of MCMC steps increases for denoising.
This was also noted by Nijkamp et al. [@nijkamp2020anatomy]. I found
that for these experiments, one has to carefully balance the amount of
MCMC steps and noise magnitude. Results for this are shown in
[19](#fig:denoising){reference-type="ref" reference="fig:denoising"} and
[21](#fig:inpainting){reference-type="ref" reference="fig:inpainting"}.

Generally, CIFAR-10 was pretty similar to MNIST, but the training time
and compute requirements were increased.

Joint energy model (JEM)
========================

Joint Energy Models (JEMs) are an idea from a paper from Grathwohl et
al. [@grathwohl2019your]. The basic idea is to take a normal
neural-network based classifier (with 'normal' meaning softmax
normalization of the logits and using the cross-entropy loss) and to use
some basic math to interpret it as an energy-based model. In this way, a
joint generative-and-classification model can be obtained, which shows
some desirable properties, such as calibration, adversarial robustness
and the ability to detect out-of-distribution samples.

The model works as follows:

We assume we have a classifier that outputs class-conditional
probabilities $p(y|\bm{x})$ using the softmax function:
$$p(y|\bm{x}) = \frac{\exp(f(\bm{x})[y])}{\sum_{y'} \exp(f(\bm{x})[y'])}$$
where $f(\bm{x})[y]$ are the logits or log-odds. Now, if we interpret
the logits to be energies $E_{\theta}(\bm{x}, y) = -f(\bm{x})[y]$, i.e.
joint energies of both $x$ and $y$, we obtain obtain a PDF of the joint
distribution using the Boltzmann distribution:
$$p(\bm{x}, y) = \frac{\exp(f(\bm{x})[y])}{Z_{\theta}}$$ Now, if we
marginalize out $y$, we get a PDF for the data $\bm{x}$:
$$p(\bm{x}) = \sum_y p(\bm{x} | y) = \frac{\sum_y \exp(f(\bm{x})[y])}{Z_{\theta}}$$
By taking the logarithm and multiplying by the partition function
$Z_\theta$, we obtain an energy for the data $\bm{x}$:
$$E_\theta(\bm{x}) = -\log \left(\sum_y \exp(f(\bm{x})[y])\right)$$

This means when we interpret the logits as joint energies, we
automatically get an energy function for the data by taking the
LogSumExp function on the logits for a given point $\bm{x}$. By jointly
training using the cross-entropy loss and the maximum likelihood
MCMC-based EBM loss, we can simultaneously train a classifier and EBM.

I implemented this idea in Tensorflow and trained it both on the MNIST
and FashionMNIST dataset. As mentioned by Grathwohl et al.
[@grathwohl2019your], the training of these models is unfortunately very
unstable. To improve stability, I noted that using more MCMC transition
steps and smoothing the target distribution using noise helped. Still,
training could be very slow and unstable.

My results where not too bad; the generated images are quite noisy
however. The classification performance was about 95% on the test set
for MNIST and 76% for FashionMNIST. The generated images are shown in
[4](#fig:jem){reference-type="ref" reference="fig:jem"}.

![Generation results for
JEM[]{label="fig:jem"}](imgs/rpbuf_samples_jem.png){#fig:jem
width="\textwidth"}

![Generation results for JEM[]{label="fig:jem"}](imgs/fm-imgs.png){#fig:jem
width="\textwidth"}

I also assessed the out-of-distribution detection performance for both
models, using different scores. The authors [@du2019implicit] propose
using the either the gradient magnitude, confidence (max probability of
any class) and energy as scores. They justify the gradient magnitude
using the asymptotic equipartition property; the idea is, that the
distribution in the typical set would be approximately uniform and thus
have low gradient magnitude and be much less uniform in the non-typical
set. This should avoid spurious modes. I used the other dataset (MNIST
for FashionMNIST, FashionMNIST for MNIST) as out-of-distribution
samples. For MNIST, the likelihood had an AUCROC of 0.97, the confidence
had an AUCROC of 0.86 and the gradient magnitude had an AUCROC of 0.86.

For FashionMNIST, the likelihood had an AUCROC of 0.86, the confidence
had an AUCROC of 0.5 (which is no better than random) and the gradient
magnitude had an AUCROC of 0.9. Generally, these are quite good results;
out-of-distribution detection can be a hard task for many generative
models and classifiers and JEM gives a considerable boost on this task.

Appendix
========

![Toy distribution:
Squares[]{label="fig:toy-squares"}](imgs/checker.png){#fig:toy-squares
width="\textwidth"}

![Toy distribution:
Squares[]{label="fig:toy-squares"}](imgs/checker_energy.png){#fig:toy-squares
width="\textwidth"}

![Toy distribution:
Squares[]{label="fig:toy-squares"}](imgs/checker_boltzmann.png){#fig:toy-squares
width="\textwidth"}

![Toy distribution:
Squares[]{label="fig:toy-squares"}](imgs/checker_samples.png){#fig:toy-squares
width="\textwidth"}

![Toy distribution:
Circles[]{label="fig:toy-circles"}](imgs/rings_samples.png){#fig:toy-circles
width="\textwidth"}

![Toy distribution:
Circles[]{label="fig:toy-circles"}](imgs/rings_energy.png){#fig:toy-circles
width="\textwidth"}

![Toy distribution:
Circles[]{label="fig:toy-circles"}](imgs/rings_boltzmann.png){#fig:toy-circles
width="\textwidth"}

![Toy distribution:
Circles[]{label="fig:toy-circles"}](imgs/rings_energy_samples.png){#fig:toy-circles
width="\textwidth"}

![Toy distribution:
Spiral[]{label="fig:toy-spiral"}](imgs/spiral.png){#fig:toy-spiral
width="\textwidth"}

![Toy distribution:
Spiral[]{label="fig:toy-spiral"}](imgs/energy_spiral.png){#fig:toy-spiral
width="\textwidth"}

![Toy distribution:
Spiral[]{label="fig:toy-spiral"}](imgs/spiral_boltzmann.png){#fig:toy-spiral
width="\textwidth"}

![Toy distribution:
Spiral[]{label="fig:toy-spiral"}](imgs/energy_spiral_samples.png){#fig:toy-spiral
width="\textwidth"}

![Denoising results for
CIFAR[]{label="fig:denoising"}](imgs/noisy-images.png){#fig:denoising
width="\textwidth"}

![Denoising results for
CIFAR[]{label="fig:denoising"}](imgs/denoised-images.png){#fig:denoising
width="\textwidth"}

![Inpainting results for
CIFAR[]{label="fig:inpainting"}](imgs/corrupted-images.png){#fig:inpainting
width="\textwidth"}

![Inpainting results for
CIFAR[]{label="fig:inpainting"}](imgs/inpainted-images-denoised.png){#fig:inpainting
width="\textwidth"}
