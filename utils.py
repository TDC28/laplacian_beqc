import numpy as np


def prepare_v_vector(nqs, v, deltas=None):
    r"""Prepares the normalized vector of function values used in the block encoding.

    Args:
        nqs (list[int]): Number of qubits for each dimension.
        v (Callable[..., float]): Function to block encode.
        deltas (list[float]): Grid spacings for each dimensions.

    Returns:
        array(float): Function values reshaped into a vector.
    """
    if deltas is None:
        deltas = [2**-nq for nq in nqs]

    points = [np.arange(0, deltas[i] * 2 ** nqs[i], deltas[i]) for i in range(len(nqs))]

    axes = np.meshgrid(*points)
    vs = v(*axes)
    vs_flat = vs.flatten()
    norm = np.linalg.norm(vs_flat)

    return vs_flat / norm


def convert_vector_to_tensor(nqs, v_vec):
    r"""Reshapes vector of values to its N-dimensional shape.

    Args:
        nqs (list[int]): Number of qubits for each dimension.
        v_vec (array(flaot)): Vector of values.

    Returns:
        array(float): Function values reshaped into a tensor.
    """
    n_points = 2 ** (np.array(nqs))

    return np.reshape(v_vec, n_points)
