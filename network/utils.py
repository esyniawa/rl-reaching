import numpy as np


def create_state_space(x_bound: tuple[int, int],
                       y_bound: tuple[int, int],
                       step_size_x: float,
                       step_size_y: float,
                       resolution_factor=1.0):
    """
    Create a state space grid based on the given bounds and step sizes.

    :param x_bound: Tuple containing the lower and upper bounds for the x-axis.
    :param y_bound: Tuple containing the lower and upper bounds for the y-axis.
    :param step_size_x: Step size for the x-axis.
    :param step_size_y: Step size for the y-axis.
    :param resolution_factor: Factor to adjust the step sizes for higher resolution. Defaults to 1.0.

    :returns: 3D array representing the state space grid with shape (len(y), len(x), 2).

    Example:
    create_state_space(x_bound=(0, 10), y_bound=(0, 5), step_size_x=1.0, step_size_y=0.5, resolution_factor=2.0)

    This function creates a grid of points in the state space based on the input bounds and step sizes.
    """

    x_lowerbound, x_upperbound = x_bound
    y_lowerbound, y_upperbound = y_bound

    # TODO implement this with linspace
    x = np.arange(start=x_lowerbound, stop=x_upperbound+step_size_x, step=resolution_factor * step_size_x)
    y = np.arange(start=y_lowerbound, stop=y_upperbound+step_size_y, step=resolution_factor * step_size_y)

    xx, yy = np.meshgrid(x, y)
    xy = np.dstack((xx, yy))

    return xy


def gauss(mu: float,
          sigma: float,
          x: np.ndarray,
          norm: bool = False,
          limit: float | None = None):

    """
    Generate a Gaussian function.

    Parameters:
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :param x: State space.
    :param norm: Normalize the density to 1 . Defaults to False.
    :param limit: Value below which the density is set to 0.
    """

    ret = np.exp(-np.power((x-mu) / sigma, 2) / 2)

    if norm:
        ret /= (np.sqrt(2.0 * np.pi) * sigma)

    if limit is not None:
        ret[ret < limit] = 0.0

    return ret


def circ_gauss(mu: float,
               sigma: float,
               n: int,
               scal: float = 1.0,
               norm: bool = False,
               limit: float | None = None,
               plot: bool = False):
    """
    Generate a circular Gaussian function.

    Parameters:
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :param n: Number of points to generate.
    :param scal: Scaling factor of the points.
    """

    mu /= scal
    ret = np.zeros(n)
    for i in range(n):
        res = abs(i-mu)
        if res > n/2:
            ret[i] = n - res
        else:
            ret[i] = res

    ret = np.exp(-np.power(scal * ret / sigma, 2) / 2)

    if norm:
        ret /= (np.sqrt(2.0 * np.pi) * sigma)

    if limit is not None:
        ret[ret < limit] = 0.0

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ret)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return ret


def multivariate_gauss(x: np.ndarray,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       norm: bool = False) -> np.ndarray:
    """
    Calculate the multivariate Gaussian density function for a given set of points.

    :param x: The points at which to evaluate the density function (N x N).
    :param mu: The mean vector of the Gaussian distribution (N).
    :param sigma: The covariance matrix of the Gaussian distribution (N x N).
    :param norm: Flag indicating whether to normalize the density function (default is False).

    :return: The value of the multivariate Gaussian density function evaluated at the given points (N x N).

    :raises ValueError: If the dimensions of mu and sigma are not compatible or if the sigma matrix is not positive definite.
    """

    # check if dimensions are correct
    dim = mu.shape[0]
    if dim != sigma.shape[0] or dim != sigma.shape[1]:
        raise ValueError("Mu must be a vector with the dimensions n x 1 and "
                         "sigma must be a matrix with dimensions n x n.")

    det_sigma = np.linalg.det(sigma)

    if np.all(np.linalg.eigvals(sigma) > 0):
        inv_sigma = np.linalg.inv(sigma)
    else:
        raise ValueError("Sigma matrix must be positive definite.")

    exp = np.einsum('...k,kl,...l->...', x - mu, inv_sigma, x - mu)

    if norm:
        return np.exp(-0.5 * exp) / np.sqrt((2*np.pi) ** dim * det_sigma)
    else:
        return np.exp(-0.5 * exp)


def bivariate_gauss(mu: tuple[float, float] | np.ndarray,
                    sigma: float,
                    xy: np.ndarray,
                    norm: bool = False, plot: bool = False, limit: float | None = None):
    """
    Calculate the bivariate Gaussian density function for a given set of points.

    :param mu: The mean vector of the Gaussian distribution (tuple of floats representing the means along each dimension).
    :param sigma: The variance of the Gaussian distribution (float).
    :param xy: The points at which to evaluate the density function (N x N numpy array).
    :param norm: Flag indicating whether to normalize the density function (default is False).
    :param plot: Flag indicating whether to plot the density function (default is False).
    :param limit: Optional threshold value for the density function (default is None).

    :return: The value of the bivariate Gaussian density function evaluated at the given points (N x N numpy array).

    :raises ValueError: If the dimensions of mu are not compatible or if the sigma value is not positive.
    """
    if isinstance(mu, tuple):
        mu = np.array(mu)

    a = multivariate_gauss(xy, mu=mu, sigma=sigma * np.eye(mu.shape[0]), norm=norm)

    if limit is not None:
        a[a < limit] = np.NaN

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = ax.contourf(a, cmap='Purples')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(img)
        plt.show()

    return a
