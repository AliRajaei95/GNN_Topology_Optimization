# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:17:59 2017
@author: jlc516
"""

import numpy as np
import time
import numpy.linalg as linalg
from scipy.stats import norm
import warnings


def sampleruniform(loads):
    """
    Input is a list of loads.
    Output is a matrix of load sets.
    The loads are sampled with +-50% variation of each single load.
    """
    matrixmulti = np.ones(shape=(len(loads), 2 * len(loads) + 1))
    for i in range(len(loads) * 2 + 1):
        if i >= 1 and i % 2 != 0:
            matrixmulti[int(i / 2), i] = 1.5
        elif i >= 1 and i % 2 == 0:
            matrixmulti[int((i - 1) / 2), i] = 0.5
    loadsm = np.multiply(np.repeat(loads[:, np.newaxis], 2 * len(loads) + 1, 1), matrixmulti)
    return loadsm


def samplersteps(loads, sampletheloads, steps):
    """
    Input is a list of loads.
    Output is a matrix of load sets.
    All loads are sampled in steps 0:0.2:2.
    """
    numsamples = pow(len(steps), len(sampletheloads))
    matrixmulti = np.ones(shape=(len(loads), numsamples))
    for j in range(len(sampletheloads)):
        makestepat = pow(len(steps), len(sampletheloads) - j - 1)
        counter = 0
        currentstep = 0
        for i in range(numsamples):
            matrixmulti[sampletheloads[j], i] = steps[currentstep]
            counter = counter + 1
            if counter >= makestepat:
                counter = 0
                currentstep = currentstep + 1
                if currentstep > len(steps) - 1:
                    currentstep = 0

    loadsm = np.multiply(np.repeat(loads[:, np.newaxis], numsamples, 1), matrixmulti)
    return loadsm


def samplermontecarlo(LB, UB, numbersamples):
    UBLB = UB - LB

    if np.size(LB) == 1:
        MLB = np.repeat(LB, numbersamples)
        MUBLB = np.repeat(UBLB, numbersamples)
    else:
        MLB = np.repeat(LB[:, np.newaxis], numbersamples, 1)
        MUBLB = np.repeat(UBLB[:, np.newaxis], numbersamples, 1)
    MCM = MLB + np.multiply(np.random.rand(np.size(UB), numbersamples), MUBLB)
    return MCM


def kumaraswamy_montecarlo(a, b, c, lower_bounds, upper_bounds, num_samples):
    """
    Perform Monte Carlo sampling of correlated samples following a Kumaraswamy distribution of parameters `a` and
    `b`. The value of `c` indicates the correlation coefficient between the sampled values.

    The bounds `LB` and `UB` must be specified for every variable i.e. the units that will be correlated. More
    correlated samples can be obtained with `num_samples`.

    :param a: The shape parameter 'a' of the Kumaraswamy distribution.
    :type a: float
    :param b: The shape parameter 'b' of the Kumaraswamy distribution.
    :type b: float
    :param c: The correlation coefficient 'c' for generating correlated samples.
    :type c: float
    :param lower_bounds: The lower bounds for each variable. Note: at least two variables are needed to perform correlation.
    :type lower_bounds: array-like
    :param upper_bounds: The upper bounds for each variable. Must have the same shape as `LB`.
    :type upper_bounds: array-like
    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :return: Matrix of Monte Carlo samples generated from the Kumaraswamy distribution.
    :rtype: ndarray

    :references:
        - Kumaraswamy distribution: https://en.wikipedia.org/wiki/Kumaraswamy_distribution
        - Cholesky decomposition: https://en.wikipedia.org/wiki/Cholesky_decomposition
        - Monte Carlo sampling: https://en.wikipedia.org/wiki/Monte_Carlo_method

    """
    # find the number of variables based on the length of the lower bounds
    num_variables = len(lower_bounds)

    # raise an error if there is only one variable
    if num_variables < 2:
        raise ValueError(f"Unexpected number of variables {num_variables}. Please use least two variables.")

    # check the dimension of UB
    if len(upper_bounds) != num_variables:
        raise ValueError(f"Dimensions of `UB` ({len(upper_bounds)}) and `LB` ({num_variables}) do not match.")

    # convert LB to ndarrays if this is not the case
    if not isinstance(lower_bounds, np.ndarray):
        warnings.warn(f"`LB` is not of type {type(np.ndarray)}. Attempting conversion", RuntimeWarning)
        lower_bounds = np.array(lower_bounds)

    # convert UB to ndarrays if this is not the case
    if not isinstance(upper_bounds, np.ndarray):
        warnings.warn(f"`UB` is not of type {type(np.ndarray)}. Attempting conversion", RuntimeWarning)
        upper_bounds = np.array(upper_bounds)

    # create a lower bound matrix (MLB) and a difference matrix (MUBLB)
    MLB = np.repeat(lower_bounds[:, np.newaxis], num_samples, 1)
    UBLB = upper_bounds - lower_bounds
    MUBLB = np.repeat(UBLB[:, np.newaxis], num_samples, 1)

    # sample the uncorrelated variables using standard normal distribution
    uncorrelated = np.random.standard_normal((num_variables, num_samples))

    # compute the covariance matrix using the provided coefficient
    cov = c * np.ones(shape=(num_variables, num_variables)) + (1 - c) * np.identity(num_variables)

    # perform Cholesky decomposition of the covariance matrix to obtain the lower triangular matrix `L`.
    L = linalg.cholesky(cov)

    # generate correlated samples
    correlated = np.dot(L, uncorrelated)

    # transform the correlated samples into a normal distribution
    cdf_correlated = norm.cdf(correlated)

    # transform into the Kumaraswamy distribution
    karamsy = pow((1 - pow((1 - cdf_correlated), (1 / b))), (1 / a))

    # scale the resulting distribution to obtain the desired boundaries
    MCM = MLB + np.multiply(karamsy, MUBLB)

    return MCM


def beta(a, b, num_samples):
    samples = np.random.beta(a, b, size=num_samples)
    return samples

# if __name__ == "__main__":
#     testresults = kumaraswamy_montecarlo(a=2, b=2, c=0.9, lower_bounds=[0], upper_bounds=[1.5], num_samples=4)