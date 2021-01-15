import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


def generate_swiss_roll(batch_size):
    data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
    data = data.astype("float32")[:, [0, 2]]
    data /= 5
    return data


def generate_circle_dataset(n_train):
    circles_train = sklearn.datasets.make_circles(n_samples=n_train, factor=.5, noise=0.08)[0]
    circles_train = circles_train.astype("float32")
    circles_train *= 3
    return circles_train


def generate_rings_dataset(n_train):
    rng = np.random.RandomState()
    obs = n_train
    n_train = n_train * 20
    n_samples4 = n_samples3 = n_samples2 = n_train // 4
    n_samples1 = n_train - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, we set endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    X = np.vstack([
        np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
        np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
    ]).T * 3.0
    X = util_shuffle(X, random_state=rng)

    # Add noise
    X = X + rng.normal(scale=0.08, size=X.shape)
    inds = np.random.choice(list(range(n_train)), obs)
    X = X[inds]
    return X.astype("float32")


def generate_moons_dataset(n_train):
    data = sklearn.datasets.make_moons(n_samples=n_train, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return data


def generate_gaussians_dataset(n_train):
    batch_size = n_train
    rng = np.random.RandomState()
    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                     1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for i in range(batch_size):
        point = rng.randn(2) * 0.5
        idx = rng.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset


def generate_pinwheel_dataset(n_train):
    batch_size = n_train
    rng = np.random.RandomState()
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = batch_size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(num_classes * num_per_class, 2) \
               * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    pinwheel_train = rng.permutation(np.einsum("ti,tij->tj", features, rotations))
    return pinwheel_train


def generate_spirals_dataset(n_train):
    batch_size = n_train
    n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x


def generate_checkboard_dataset(n_train):
    batch_size = n_train
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    checkerboard_train = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    return checkerboard_train


def generate_line_dataset(n_train):
    batch_size = n_train
    rng = np.random.RandomState()
    x = rng.rand(batch_size) * 5 - 2.5
    y = x
    line_train = np.stack((x, y), 1)
    return line_train


def generate_cosine_dataset(n_train):
    batch_size = n_train
    rng = np.random.RandomState()
    x = rng.rand(batch_size) * 5 - 2.5
    y = np.sin(x) * 2.5
    cosine_train = np.stack((x, y), 1)
    return cosine_train
