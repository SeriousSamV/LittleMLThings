import math
from collections import Counter
from typing import List

from linear_algebra import sum_of_squares, dot

num_friends = [100.0, 49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12, 11,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
               9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1]

daily_minutes = [1, 68.77, 51.25, 52.08, 38.36, 44.54, 57.13, 51.4, 41.42, 31.22, 34.76, 54.01, 38.79, 47.59, 49.1,
                 27.66, 41.03, 36.73, 48.65, 28.12, 46.62, 35.57, 32.98, 35, 26.07, 23.77, 39.73, 40.57, 31.65, 31.21,
                 36.32, 20.45, 21.93, 26.02, 27.34, 23.49, 46.94, 30.5, 33.8, 24.23, 21.4, 27.94, 32.24, 40.57, 25.07,
                 19.42, 22.39, 18.42, 46.96, 23.72, 26.41, 26.97, 36.76, 40.32, 35.02, 29.47, 30.2, 31, 38.11, 38.18,
                 36.31, 21.03, 30.86, 36.07, 28.66, 29.08, 37.28, 15.28, 24.17, 22.31, 30.17, 25.53, 19.85, 35.37, 44.6,
                 17.23, 13.47, 26.33, 35.02, 32.09, 24.81, 19.33, 28.77, 24.26, 31.98, 25.73, 24.86, 16.28, 34.51,
                 15.23, 39.72, 40.8, 26.06, 35.76, 34.76, 16.13, 44.04, 18.03, 19.65, 32.62, 35.59, 39.43, 14.18, 35.24,
                 40.13, 41.82, 35.45, 36.07, 43.67, 24.61, 20.9, 21.9, 18.79, 27.61, 27.21, 26.61, 29.77, 20.59, 27.53,
                 13.82, 33.2, 25, 33.1, 36.65, 18.63, 14.87, 22.2, 36.81, 25.53, 24.62, 26.25, 18.21, 28.08, 19.42,
                 29.79, 32.8, 35.99, 28.32, 27.79, 35.88, 29.06, 36.28, 14.1, 36.63, 37.49, 26.9, 18.58, 38.48, 24.48,
                 18.95, 33.55, 14.24, 29.04, 32.51, 25.63, 22.22, 19, 32.73, 15.16, 13.9, 27.2, 32.01, 29.27, 33, 13.74,
                 20.42, 27.32, 18.23, 35.35, 28.48, 9.08, 24.62, 20.12, 35.26, 19.92, 31.02, 16.49, 12.16, 30.7, 31.22,
                 34.65, 13.13, 27.51, 33.2, 31.57, 14.1, 33.42, 17.44, 10.12, 24.42, 9.82, 23.39, 30.93, 15.03, 21.67,
                 31.09, 33.29, 22.61, 26.89, 23.48, 8.38, 27.81, 32.35, 23.84]

daily_hours = [dm / 60 for dm in daily_minutes]


def mean(xs: List[float]) -> float:
    """
    Calculates the mean (average) of a list of numbers.

    This function computes the mean by summing all the numbers
    in the given list and dividing by the count of numbers in the list.
    It assumes the input list is non-empty.

    :param xs:```python A list

    >>> mean([1, 2, 3, 4, 5])
    3.0
    """
    return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    """
    Calculate the median of a list of floating-point numbers.

    This function determines whether the length of the list is even or odd.
    For an even-length list, it calculates the median using the helper function
    `_median_even`. For an odd-length list, it calculates the median using the
    helper function `_median_odd`.

    :param v: List of floating-point numbers for which to calculate the median.
    :type v: List[float]
    :return: The median value of the provided list.
    :rtype: float

    >>> median([1, 3, 5])
    3

    >>> median([1, 3, 5, 7])
    4.0
    """
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


def quantile(xs: List[float], p: float) -> float:
    """
    Calculates the pth quantile of a given list of numerical values.

    The function determines the value at the specified quantile by first
    sorting the list of numbers and then selecting the value at the
    calculated quantile index.

    :param xs: A list of numerical values for which the quantile is to be computed.
    :param p: A float value between 0 and 1 representing the desired quantile.
    :return: The value at the pth quantile in the sorted list.

    >>> quantile(num_friends, 0.10)
    1

    >>> quantile(num_friends, 0.25)
    3

    >>> quantile(num_friends, 0.90)
    13
    """
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(xs: List[float]) -> List[float]:
    """
    Calculates the mode(s) of a list of numbers. The mode is the value or values
    that appear most frequently in the list.

    :param xs: A list of floating-point numbers to find the mode(s) of.
    :type xs: List[float]

    :return: A list containing the mode(s) of the input list. If multiple
             values share the highest frequency, all of them are returned.
    :rtype: List[float]

    >>> mode([1, 2, 3])
    [1, 2, 3]

    >>> mode([1, 2, 3, 2, 1])
    [1, 2]

    >>> mode([1, 2, 3, 2, 1, 2, 1])
    [1, 2]

    >>> mode([1, 2, 3, 2, 1, 2, 1, 1, 1])
    [1]
    """
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


assert set(mode(num_friends)) == {1, 6}


def data_range(xs: List[float]) -> float:
    """
    Calculate the range of a dataset.

    This function computes the difference between the maximum and minimum
    values in a list of floating-point numbers, which represents the range
    of the data.

    :param xs: List of floating-point numbers representing the dataset.
    :type xs: List[float]
    :return: The range of the dataset, computed as the difference between
        the maximum and minimum values in the list.
    :rtype: float

    >>> data_range([1, 2, 3])
    2

    >>> data_range([1, 100])
    99
    """
    return max(xs) - min(xs)


assert data_range(num_friends) == 99


def de_mean(xs: List[float]) -> List[float]:
    """Deviation from Mean

    Compute the deviation of each element in a list from the mean of the list.

    :param xs: A list of floating-point numbers to calculate the deviations.
    :return: A list of floating-point numbers representing the deviations of the
        input elements from their mean.

    >>> de_mean([1, 2, 3])
    [-1.0, 0.0, 1.0]
    """
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


assert de_mean([1, 2, 3]) == [-1.0, 0.0, 1.0]


def variance(xs: List[float]) -> float:
    """
    Computes the variance of a sequence of numeric values. The function requires at
    least two elements in the provided sequence to calculate the variance. Variance
    measures the average squared deviation from the mean of the sequence values.

    :param xs: A list of numeric values for which the variance is computed. Must
        contain at least two elements.
    :type xs: List[float]

    :return: The variance of the input sequence, representing the average of the
        squared deviations from the mean.
    :rtype: float

    :raises AssertionError: If the input sequence has fewer than two elements.

    >>> 81.54 < variance(num_friends) < 81.55
    True

    >>> variance([1, 2, 3, 4, 5])
    2.5

    >>> variance([1, 2, 3])
    1.0
    """
    assert len(xs) >= 2

    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (len(xs) - 1)


def standard_deviation(xs: List[float]) -> float:
    """
    Calculate the standard deviation of a list of numbers.

    The standard deviation is a measure that quantifies the amount of variation
    or dispersion in a data set. It is calculated as the square root of the
    variance.

    :param xs: A list of floating-point numbers for which the standard deviation
        is to be calculated.
    :return: The calculated standard deviation as a floating-point number.

    >>> standard_deviation([1, 2, 3])
    1.0

    >>> 9.02 < standard_deviation(num_friends) < 9.04
    True
    """
    return math.sqrt(variance(xs))


def inter_quartile_range(xs: List[float]) -> float:
    """
    Returns the difference between the 75%-ile and the 25%-ile

    >>> inter_quartile_range(num_friends)
    6

    >>> inter_quartile_range([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5
    """
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def covariance(xs: List[float], ys: List[float]) -> float:
    """
    Calculate the covariance between two lists of numerical data.

    The covariance is a measure of how two variables change together. If the covariance
    is positive, it indicates that both variables increase together. If negative, one
    variable tends to decrease while the other increases. Covariance is computed using
    the de-meaned values of the inputs and their dot product, divided by the number of
    observations minus one.

    :param xs: The first list of numerical data. Its length must match the length of `ys`.
    :param ys: The second list of numerical data. Its length must match the length of `xs`.
    :return: The covariance between `xs` and `ys`.

    >>> 22.42 < covariance(num_friends, daily_minutes) < 22.43
    True

    >>> 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60
    True

    >>> covariance([1, 2, 3], [2, 3, 4])
    1.0

    >>> covariance([1, 2, 3], [3, 4, 5])
    1.0

    >>> covariance([1, 2, 3], [3, 4, -15])
    -9.0

    >>> covariance([1, 2, 3], [3, 4, 15])
    6.0

    >>> covariance([1, 2, 3], [-1, -2, -3])
    -1.0
    """
    assert len(xs) == len(ys)

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
    """
    Calculate the Pearson correlation coefficient for two lists of numerical values.

    The Pearson correlation measures the linear relationship between two datasets.
    It is computed as the covariance of two variables divided by the product of
    their standard deviations. The value of the Pearson correlation ranges from
    -1 to +1, where -1 indicates a perfect negative linear relationship, 0 indicates
    no linear relationship, and +1 indicates a perfect positive linear relationship.

    This function first calculates the standard deviations of both input datasets (`xs`
    and `ys`). If both standard deviations are greater than zero, the correlation
    is calculated as the covariance divided by the product of the standard deviations.
    If either of the standard deviations is zero, the function will return 0,
    indicating no meaningful correlation can be computed.

    :param xs: A list of numerical values representing the first dataset.
    :param ys: A list of numerical values representing the second dataset.
    :return: The Pearson correlation coefficient as a float value. If the standard
        deviation of either dataset is zero, it returns 0.

    >>> correlation([1, 2, 3], [3, 4, 5])
    1.0

    >>> 0.9 < correlation([1, 2, 3], [3, 4, 15]) < 0.909
    True

    >>> correlation([1, 2, 3], [-1, -2, -3])
    -1.0
    """
    stddev_x = standard_deviation(xs)
    stddev_y = standard_deviation(ys)
    if stddev_x > 0 and stddev_y > 0:
        return covariance(xs, ys) / stddev_x / stddev_y
    else:
        return 0
