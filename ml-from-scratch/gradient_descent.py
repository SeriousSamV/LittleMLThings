import random
from typing import Callable, List, TypeVar

from linear_algebra import Vector, distance, add, scalar_multiply, vector_mean

T = TypeVar('T')


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Calculate the difference quotient of a function at a point.

    Approximates the derivative of a function f at point x using the
    difference quotient formula: (f(x + h) - f(x)) / h

    This is a numerical approximation of the derivative, useful when
    the analytical derivative is difficult to compute.

    :param f: The function to differentiate
    :param x: The point at which to approximate the derivative
    :param h: A small step size for the approximation
    :return: The approximate derivative of f at x

    >>> def square(x): return x * x
    >>> def derivative(x): return 2 * x
    >>> x = 10
    >>> abs(difference_quotient(square, x, 0.00001) - derivative(x)) < 0.001
    True
    """
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """
    Calculate the partial derivative of f with respect to the i-th variable.

    Computes the partial difference quotient by varying only the i-th
    component of the input vector v while keeping all other components fixed.

    :param f: A function that takes a vector and returns a scalar
    :param v: The point (vector) at which to compute the partial derivative
    :param i: The index of the variable with respect to which we differentiate
    :param h: A small step size for the approximation
    :return: The approximate partial derivative of f with respect to v[i]

    >>> def sum_of_squares(v: Vector) -> float:
    ...     return sum(x ** 2 for x in v)
    >>> v = [1.0, 2.0, 3.0]
    >>> abs(partial_difference_quotient(sum_of_squares, v, 0, 0.0001) - 2.0) < 0.01
    True
    >>> abs(partial_difference_quotient(sum_of_squares, v, 1, 0.0001) - 4.0) < 0.01
    True
    """
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                     v: Vector,
                     h: float = 0.0001) -> Vector:
    """
    Estimate the gradient of f at point v using numerical differentiation.

    Computes all partial derivatives to form the gradient vector,
    which points in the direction of steepest ascent.

    :param f: A function that takes a vector and returns a scalar
    :param v: The point at which to estimate the gradient
    :param h: A small step size for numerical differentiation
    :return: The gradient vector (all partial derivatives)

    >>> def sum_of_squares(v: Vector) -> float:
    ...     return sum(x ** 2 for x in v)
    >>> v = [1.0, 2.0, 3.0]
    >>> gradient = estimate_gradient(sum_of_squares, v)
    >>> all(abs(gradient[i] - 2 * v[i]) < 0.01 for i in range(len(v)))
    True
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Move step_size in the gradient direction from v.

    :param v: Current position vector
    :param gradient: Gradient vector (direction of steepest ascent)
    :param step_size: How far to move (learning rate)
    :return: New position after taking the gradient step

    >>> gradient_step([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], -0.1)
    [0.99, 1.98, 2.97]
    """
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def gradient_descent(f: Callable[[Vector], float],
                    gradient_f: Callable[[Vector], Vector],
                    theta_0: Vector,
                    learning_rate: float = 0.01,
                    num_iterations: int = 1000,
                    tolerance: float = 0.00001) -> Vector:
    """
    Minimize a function using gradient descent.

    Starting from theta_0, iteratively takes steps in the direction of
    the negative gradient to find a local minimum of f.

    :param f: The function to minimize
    :param gradient_f: Function that computes the gradient of f
    :param theta_0: Initial starting point
    :param learning_rate: Step size for each iteration
    :param num_iterations: Maximum number of iterations
    :param tolerance: Stop if the step size becomes smaller than this
    :return: The parameters that (approximately) minimize f

    >>> def sum_of_squares(v: Vector) -> float:
    ...     return sum(x ** 2 for x in v)
    >>> def sum_of_squares_gradient(v: Vector) -> Vector:
    ...     return [2 * x for x in v]
    >>> result = gradient_descent(sum_of_squares, sum_of_squares_gradient, [5.0, 5.0, 5.0], 0.01, 10000)
    >>> all(abs(x) < 0.01 for x in result)
    True
    """
    theta = theta_0

    for iteration in range(num_iterations):
        gradient = gradient_f(theta)
        next_theta = gradient_step(theta, gradient, -learning_rate)

        # Stop if we're converging
        if distance(next_theta, theta) < tolerance:
            return next_theta

        theta = next_theta

    return theta


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    """
    Gradient of the squared error for a single training example.

    For a linear model y = theta[0] + theta[1] * x, computes the gradient
    of the squared error (prediction - actual)^2.

    :param x: Input feature value
    :param y: Actual output value
    :param theta: Model parameters [intercept, slope]
    :return: Gradient vector [d/d(theta[0]), d/d(theta[1])]
    """
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    return [2 * error, 2 * error * x]


def minibatches(dataset: List[T],
               batch_size: int,
               shuffle: bool = True) -> List[List[T]]:
    """
    Split dataset into batches of batch_size.

    Optionally shuffles the data before creating batches.
    Useful for mini-batch gradient descent.

    :param dataset: The full dataset to split
    :param batch_size: Size of each batch
    :param shuffle: Whether to shuffle before batching
    :return: List of batches

    >>> data = list(range(10))
    >>> batches = minibatches(data, 3, shuffle=False)
    >>> len(batches)
    4
    >>> len(batches[0])
    3
    >>> len(batches[-1])
    1
    """
    batch_starts = list(range(0, len(dataset), batch_size))

    if shuffle:
        random.shuffle(batch_starts)

    return [dataset[i:i + batch_size] for i in batch_starts]


def stochastic_gradient_descent(f: Callable[[T, Vector], float],
                               gradient_f: Callable[[T, Vector], Vector],
                               data: List[T],
                               theta_0: Vector,
                               learning_rate: float = 0.01,
                               num_epochs: int = 100) -> Vector:
    """
    Minimize using stochastic gradient descent.

    Instead of computing the gradient over the entire dataset,
    updates parameters using one example at a time. This can be
    much faster for large datasets.

    :param f: Loss function for a single data point
    :param gradient_f: Function that computes gradient for a single data point
    :param data: List of training examples
    :param theta_0: Initial parameters
    :param learning_rate: Step size for updates
    :param num_epochs: Number of passes through the data
    :return: The parameters that minimize the loss

    >>> # Example: fit y = 3x + 5 with noise
    >>> random.seed(42)
    >>> data = [(x, 3 * x + 5 + random.gauss(0, 0.1)) for x in range(100)]
    >>> def loss(xy, theta):
    ...     x, y = xy
    ...     return (theta[0] * x + theta[1] - y) ** 2
    >>> def grad(xy, theta):
    ...     x, y = xy
    ...     error = theta[0] * x + theta[1] - y
    ...     return [2 * error * x, 2 * error]
    >>> theta = stochastic_gradient_descent(loss, grad, data, [0.0, 0.0], 0.001, 1000)
    >>> abs(theta[0] - 3.0) < 0.1  # slope should be close to 3
    True
    >>> abs(theta[1] - 5.0) < 0.1  # intercept should be close to 5
    True
    """
    theta = theta_0

    for epoch in range(num_epochs):
        for example in data:
            gradient = gradient_f(example, theta)
            theta = gradient_step(theta, gradient, -learning_rate)

    return theta


def minibatch_gradient_descent(f: Callable[[List[T], Vector], float],
                              gradient_f: Callable[[List[T], Vector], Vector],
                              data: List[T],
                              theta_0: Vector,
                              learning_rate: float = 0.01,
                              batch_size: int = 32,
                              num_epochs: int = 100) -> Vector:
    """
    Minimize using mini-batch gradient descent.

    A compromise between batch gradient descent (uses all data)
    and stochastic gradient descent (uses one example). Updates
    parameters using small batches of data.

    :param f: Loss function for a batch of data
    :param gradient_f: Function that computes gradient for a batch
    :param data: List of training examples
    :param theta_0: Initial parameters
    :param learning_rate: Step size for updates
    :param batch_size: Number of examples per batch
    :param num_epochs: Number of passes through the data
    :return: The parameters that minimize the loss

    >>> # Example: fit y = 2x + 1
    >>> random.seed(42)
    >>> data = [(x, 2 * x + 1 + random.gauss(0, 0.1)) for x in range(100)]
    >>> def batch_loss(batch, theta):
    ...     return sum((theta[0] * x + theta[1] - y) ** 2 for x, y in batch) / len(batch)
    >>> def batch_grad(batch, theta):
    ...     grad = [0.0, 0.0]
    ...     for x, y in batch:
    ...         error = theta[0] * x + theta[1] - y
    ...         grad[0] += 2 * error * x
    ...         grad[1] += 2 * error
    ...     return [g / len(batch) for g in grad]
    >>> theta = minibatch_gradient_descent(batch_loss, batch_grad, data, [0.0, 0.0], 0.01, 10, 1000)
    >>> abs(theta[0] - 2.0) < 0.1
    True
    >>> abs(theta[1] - 1.0) < 0.1
    True
    """
    theta = theta_0

    for epoch in range(num_epochs):
        batches = minibatches(data, batch_size, shuffle=True)
        for batch in batches:
            gradient = gradient_f(batch, theta)
            theta = gradient_step(theta, gradient, -learning_rate)

    return theta


if __name__ == '__main__':
    # Example: Minimize sum of squares
    print("=" * 60)
    print("Example 1: Minimize sum of squares using gradient descent")
    print("=" * 60)

    def sum_of_squares(v: Vector) -> float:
        return sum(x ** 2 for x in v)

    def sum_of_squares_gradient(v: Vector) -> Vector:
        return [2 * x for x in v]

    # Start at a random point
    starting_point = [random.uniform(-10, 10) for _ in range(3)]
    print(f"Starting point: {starting_point}")
    print(f"Initial value: {sum_of_squares(starting_point):.4f}")

    result = gradient_descent(
        sum_of_squares,
        sum_of_squares_gradient,
        starting_point,
        learning_rate=0.01,
        num_iterations=1000
    )

    print(f"Minimum found at: {result}")
    print(f"Minimum value: {sum_of_squares(result):.6f}")
    print()

    # Example: Fit a linear model using stochastic gradient descent
    print("=" * 60)
    print("Example 2: Fit y = 3x + 5 using SGD")
    print("=" * 60)

    random.seed(42)
    # Generate data: y = 3x + 5 with some noise
    true_slope, true_intercept = 3.0, 5.0
    data_points = [(x, true_slope * x + true_intercept + random.gauss(0, 0.5))
                   for x in range(50)]

    def loss(xy, theta):
        x, y = xy
        predicted = theta[0] * x + theta[1]
        return (predicted - y) ** 2

    def grad(xy, theta):
        x, y = xy
        predicted = theta[0] * x + theta[1]
        error = predicted - y
        return [2 * error * x, 2 * error]

    print(f"True parameters: slope={true_slope}, intercept={true_intercept}")

    fitted_params = stochastic_gradient_descent(
        loss,
        grad,
        data_points,
        [0.0, 0.0],  # Start with zeros
        learning_rate=0.0001,
        num_epochs=100
    )

    print(f"Fitted parameters: slope={fitted_params[0]:.4f}, intercept={fitted_params[1]:.4f}")
    print()

    # Example: Using numerical gradient estimation
    print("=" * 60)
    print("Example 3: Minimize x^2 + y^2 + z^2 using numerical gradients")
    print("=" * 60)

    starting_point = [5.0, -3.0, 8.0]
    print(f"Starting point: {starting_point}")

    result_numerical = gradient_descent(
        sum_of_squares,
        lambda v: estimate_gradient(sum_of_squares, v),
        starting_point,
        learning_rate=0.01,
        num_iterations=1000
    )

    print(f"Minimum found at: {result_numerical}")
    print(f"Minimum value: {sum_of_squares(result_numerical):.6f}")
