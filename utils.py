"""
Utility functions for data input/output, converting quantum circuits to unitary matrices, and plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from block_encoding import generate_laplacian_block_encoding


def prepare_v_vector(nqs, v, deltas=None):
    r"""Prepares the normalized vector of function values used in the block encoding.

    Args:
        nqs (list[int]): Number of qubits for each dimension.
        v (Callable[..., float]): Function to block encode.
        deltas (list[float]): Grid spacings for each dimensions.

    Returns:
        ndarray(float): Function values reshaped into a vector.
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
        ndarray(float): Function values reshaped into a tensor.
    """
    n_points = 2 ** (np.array(nqs))

    return np.reshape(v_vec, n_points)


def get_circuit_unitary(qc, nqs, subspace=True):
    r"""Build the matrix representation of a quantum circuit block encoding a Laplacian.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit that block encodes a Laplacian.
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.

    Returns:
        ndarray(float): Representation of the block encoded Laplacian matrix.
    """
    simulator = AerSimulator(method="unitary")
    qc = transpile(qc, simulator)

    result = simulator.run(qc).result()
    unitary = result.get_unitary(qc).data.real

    if subspace:
        unitary_subspace = unitary[: 2 ** sum(nqs), : 2 ** sum(nqs)]

        return unitary_subspace

    return unitary


def plot_heatmap(nqs, deltas=None, bcs=None, ncs=101, vmax=1.0):
    r"""Plot the matrix elements for a Laplacian operator.

    Args:
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.
        deltas (list[float]): Grid spacings for each dimension.
        bcs (list[str]): Boundary conditions of the Laplacian.
        ncs (int): Number of colors to use in the plot.
        vmax (float): Color scaling set to -vmax to vmax.
    """
    qc = generate_laplacian_block_encoding(nqs, deltas, bcs)
    unitary = get_circuit_unitary(qc, nqs)
    cmap = plt.get_cmap("seismic", ncs)

    plt.imshow(unitary, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.show()
