import numpy as np
import scipy.linalg


def inception_score(p_yx):
    p_y = np.mean(p_yx, axis=0)
    d_kl = np.sum(p_yx * (np.log(p_yx + 1e-6) - np.log(p_y + 1e-6)), axis=1)
    return np.exp(np.mean(d_kl))


def frechet_inception_distance(statistics_gen, statistics_real):
    mu_g = np.mean(statistics_gen, axis=0)
    cov_g = np.cov(statistics_gen)
    mu_r = np.mean(statistics_real, axis=0)
    cov_r = np.cov(statistics_real)
    diff_mean = np.sum((mu_g - mu_r) ** 2)
    cov_mean_geom = scipy.linalg.sqrtm(cov_r * cov_g)
    if np.iscomplexobj(cov_mean_geom):
        cov_mean_geom = cov_mean_geom.real
    return diff_mean + np.trace(cov_g + cov_r - 2 * cov_mean_geom)
