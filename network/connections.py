import numpy as np


def column_wise_connection(preDim: int,
                           postDim: tuple[int, int] | list[int, int],
                           weight=1.0):

    i, j = postDim
    w = np.array([[[None]*preDim]*j]*i)

    for pre_i in range(preDim):
        w[:, pre_i, pre_i] = weight

    return w.reshape(i*j, preDim)


def row_wise_connection(preDim: int,
                        postDim: tuple[int, int] | list[int, int],
                        weight=1.0):

    i, j = postDim
    w = np.array([[[None]*preDim]*j]*i)

    for pre_i in range(preDim):
        w[pre_i, :, pre_i] = weight

    return w.reshape(i*j, preDim)


def w_2D_to_3D_S1(preDim: list | tuple, postDim: list | tuple, weight: float = 1.0):

    if isinstance(preDim, tuple):
        preDim = list(preDim)
    if isinstance(postDim, tuple):
        postDim = list(postDim)

    post_k = postDim.pop(-1)
    assert np.array(preDim).shape == np.array(postDim).shape, "Remaining Dimensions should align!"

    n = np.prod(preDim)
    w = np.array([[[None]*n]*post_k]*n)

    for i in range(n):
        w[i, :, i] = weight

    return w.reshape((n*post_k, n))


def w_ones_to_all(preDim: list | tuple, postDim: list | tuple, allDim: int = -1, weight: float = 1.0):
    """
    Returns connection matrix, where all layers are connected one-to-one, except one layer (allDim), which is connected
    all-to-all.
    :param preDim: Shape of pre-synaptic layer
    :param postDim: Shape of post-synaptic layer
    :param allDim:  Index of the layer on which the all-to-all connection takes place.
    :param weight: Connection strength
    :return:
    """
    if isinstance(preDim, tuple):
        preDim = list(preDim)
    if isinstance(postDim, tuple):
        postDim = list(postDim)

    pre_k, post_k = preDim.pop(allDim), postDim.pop(allDim)

    if not preDim == postDim:
        raise AssertionError

    n = np.prod(preDim)
    w = np.array([[[[None] * post_k] * n] * pre_k] * n)

    for i in range(n):
        for k_pre in range(pre_k):
            for k_post in range(post_k):
                w[i, k_pre, i, k_post] = weight

    return w.reshape((n * pre_k, n * post_k)).T


def w_one_to_ones(preDim: int, postDim: tuple | list, postIndex: int = -1, weight: float = 1.0):

    if isinstance(postDim, tuple):
        postDim = list(postDim)

    post_k = postDim.pop(postIndex)
    assert post_k == preDim, "Remaining Dimensions should align!"

    n = np.prod(postDim)
    w = np.array([[[None]*preDim]*post_k]*n)

    for i in range(n):
        for k_post in range(post_k):
            w[i, k_post, k_post] = weight

    return w.reshape((n*post_k, preDim))


def w_ones_to_one(preDim: tuple, postDim: int, postIndex: int = -1, weight: float = 1.0):
    pass


def w_pooling(preDim: tuple | list, poolingDim: int = -1, weight: float = 1.0):
    """
    Just a summation pooling connection. Implemented because ANNarchys Pooling Function doesn't support summation.
    :param preDim: Shape of pre-synaptic layer
    :param poolingDim: Index of the layer on which summation takes place.
    :param weight: Connection strength
    :return:
    """

    if isinstance(preDim, tuple):
        preDim = list(preDim)

    pre_k = preDim.pop(poolingDim)

    n = np.prod(preDim)
    w = np.array([[[None] * pre_k] * pre_k] * n)

    for i in range(n):
        for k_pre in range(pre_k):
            w[i, k_pre, k_pre] = weight

    return w.reshape((n * pre_k, pre_k)).T


def pop_code_output(preferred_angles: np.ndarray, radians: bool = False):
    """
    This matrix reads out a population code in which neurons prefer a certain angle in space. The first neuron encodes
    the resulting movement angles from the population activity, the second the x component and the third the y component
    of the movement. The movement angle must also be normalized (not part of this connection matrix).
    :param preferred_angles: Preferred angle of the neurons
    :param radians: Specification of the angles in radians?
    :return w: Resulting connectivity matrix
    """

    pre_shape = preferred_angles.shape[0]
    w = np.array([[None]*pre_shape]*3)

    # angles
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[0, i] = float(np.degrees(angle))
        else:
            w[0, i] = float(angle)

    # x component
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[1, i] = np.cos(angle)
        else:
            w[1, i] = np.cos(np.radians(angle))

    # y component
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[2, i] = np.sin(angle)
        else:
            w[2, i] = np.sin(np.radians(angle))

    return w


def laterals_layerwise(Dim, axis,
                       weight: float = 1.0,
                       subset_dict: dict | None = None,
                       allow_self_con: bool = False):
    """
    Create lateral connections between layers in a neural network.

    Parameters:
    - Dim (tuple): The dimensions of the neural network layers.
    - axis (int): The axis along which lateral connections will be created.
    - weight (float, optional): The weight of the lateral connections (default is 1.0).
    - allow_self_con (bool, optional): Whether to allow self-connections (default is False).

    Returns:
    - numpy.ndarray: The lateral connection weights reshaped for the neural network.

    """

    if isinstance(Dim, tuple):
        Dim = list(Dim)

    new_dim = Dim.copy()
    pre_ = new_dim.pop(axis)

    if subset_dict is not None:
        subsets = []
        for key in subset_dict:
            subsets.append(np.arange(subset_dict[key][0], subset_dict[key][1]))
    else:
        subsets = [np.arange(0, pre_)]

    n = np.prod(new_dim)
    new_dim += [pre_]

    w = np.array([[[[None] * pre_] * n] * pre_] * n)

    for layer in range(n):
        for subset in subsets:
            for i in subset:
                for j in subset:
                    if i != j:
                        w[layer, i, layer, j] = weight

    w = w.reshape(new_dim + new_dim)
    w = np.moveaxis(w, len(Dim)-1, axis)
    w = np.moveaxis(w, -1, axis+len(Dim))
    w = w.reshape((np.prod(Dim), np.prod(Dim)))

    return w


def connect_gaussian_circle(Dim: int, sd: float, scale: float,
                            A: float = 1.0, limit: float | None = 0.001):

    from .utils import circ_gauss

    w = np.zeros((Dim, Dim))

    for i in range(Dim):
        w[i, :] = A * circ_gauss(mu=i*scale, sigma=sd, scal=scale, n=Dim, norm=False, limit=limit)

    return w.T
