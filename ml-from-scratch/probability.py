import enum
import math
import random


class Kid(enum.Enum):
    BOY = 1
    GIRL = 0


def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])


if __name__ == '__main__':
    both_girls = 0
    both_boys = 0
    either_boy = 0
    older_boy = 0

    random.seed(42)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == Kid.BOY:
            older_boy += 1
        if older == Kid.BOY or younger == Kid.BOY:
            either_boy += 1
        if younger == Kid.BOY and older == Kid.BOY:
            both_boys += 1
        if younger == Kid.GIRL and older == Kid.GIRL:
            both_girls += 1

    print("P(both | older) is a boy: ", both_boys / older_boy)
    print("P(both | either) is a boy: ", both_boys / either_boy)


def uniform_pdf(x: float) -> float:
    """
    Probability density for a uniform distribution.

    This function represents the probability density function (PDF) of a
    continuous uniform distribution on the interval [0, 1). For any input
    value `x` within this interval, the function returns a constant value
    of 1. For values outside the interval, the function returns 0.

    :param x: A floating-point number for which the uniform distribution PDF will be evaluated.
    :return: The probability density of the uniform distribution at the input `x`. Returns 1 if `x` is
        in the interval [0, 1), and 0 otherwise.
    """
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """
    Cumulative Distribution Function (CDF)

    Computes the cumulative distribution function (CDF) for a uniform
    distribution defined over the range [0, 1]. The function returns the
    probability that a value drawn from the uniform distribution is less
    than or equal to the given input.

    :param x: The input value for which the CDF is to be calculated.
    :return: The computed CDF value.

    >>> uniform_cdf(0.5)
    0.5

    >>> uniform_cdf(1.5)
    1

    >>> uniform_cdf(-0.5)
    0

    >>> uniform_cdf(2.0)
    1
    """
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probability Density Function (PDF) of a Normal Distribution

    Calculates the value of the probability density function (PDF) of a normal (Gaussian)
    distribution for a given input `x` with specified mean `mu` and standard deviation `sigma`.

    This function uses the formula:

        PDF(x) = (1 / (sqrt(2 * pi) * sigma)) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    This formula computes the likelihood of `x` occurring in a normal distribution
    defined by `mu` (mean) and `sigma` (standard deviation).

    :param x: The input value for which the PDF is evaluated
    :param mu: The mean (expected value) of the normal distribution
    :param sigma: The standard deviation (dispersion) of the normal distribution
    :return: The computed probability density function value at `x`
    :rtype: float
    """
    return math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Cumulative Distribution Function (CDF) of a Normal Distribution

    Calculates the cumulative distribution function (CDF) for a value ``x``
    of a normal distribution with mean ``mu`` and standard deviation ``sigma``.
    This function computes the probability that a randomly chosen value from
    the specified normal distribution is less than or equal to ``x``.

    :param x:
        The value for which the cumulative distribution is calculated.
    :param mu:
        The mean of the normal distribution. Defaults to 0.
    :param sigma:
        The standard deviation of the normal distribution. Defaults to 1.
    :return:
        The probability that a random variable from the specified normal
        distribution is less than or equal to the given value ``x``.

    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
    """
    Inverse CDF of a Normal Distribution

    Computes the inverse of the cumulative distribution function (CDF) of a normal
    distribution. This function effectively finds the z-score that corresponds to the
    given probability `p` under a normal distribution specified by the mean (`mu`) and
    standard deviation (`sigma`). The computation uses binary search iteration
    to approximate the value up to the provided tolerance.

    If the normal distribution has a non-standard mean (`mu`) or standard deviation
    (`sigma`), a transformation is applied to normalize the distribution before finding
    the inverse CDF value.

    :param p: The target probability (between 0 and 1) for which to compute the
        corresponding z-score.
    :param mu: The mean of the normal distribution. Defaults to 0.
    :param sigma: The standard deviation of the normal distribution. Defaults to 1.
    :param tolerance: The precision tolerance for the result. The iteration stops
        once the difference between high and low bounds is smaller than this value.
    :return: The z-score corresponding to the given probability `p` under the normal
        distribution specified by `mu` and `sigma`.
    :rtype: float
    """
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10.0
    mid_z = 0.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z
