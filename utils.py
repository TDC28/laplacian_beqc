"""
Utility functions for data input/output, converting quantum circuits to unitary matrices, and plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from qiskit import transpile
from qiskit_aer import AerSimulator


# Manually computed Laplacian matrices
def lap1d_fd(n, h=1.0, bc="dirichlet"):
    """
    Build 1D second-order finite-difference  (standard 3-point centered (central) finite-difference discretization for the second derivative)
    Laplacian for `n` grid points with spacing `h` and boundary condition `bc`.

    Parameters
    ----------
    n : int
        Number of grid points (nodes) along the 1D domain (>=1).
    h : float
        Grid spacing.
    bc : {'dirichlet','neumann','periodic'}
        Boundary condition type:
          - 'dirichlet' : u=0 at boundaries (standard tridiagonal).
          - 'neumann'   : zero-flux (one-sided treatment at endpoints).
          - 'periodic'  : wrap-around neighbors (circulant/tridiagonal with corners).

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n,n)
        Sparse 1D Laplacian matrix (second derivative approximation) scaled by 1/h^2.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    main = -2.0 / (h * h)
    off = 1.0 / (h * h)

    # tridiagonal base (LIL for easy element assignment)
    A = sp.diags(
        [off * np.ones(n - 1), main * np.ones(n), off * np.ones(n - 1)],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="lil",
    )

    if bc == "periodic":
        # wrap-around entries
        if n >= 2:
            A[0, n - 1] = off
            A[n - 1, 0] = off
        else:
            # n == 1: Laplacian on single periodic point is zero
            A[0, 0] = 0.0

    elif bc == "neumann":
        if n == 1:
            A[0, 0] = 0.0
        else:
            # Neumann BC
            A[0, 0] = -1.0 / (h * h)
            A[0, 1] = 1.0 / (h * h)

            A[n - 1, n - 2] = 1.0 / (h * h)
            A[n - 1, n - 1] = -1.0 / (h * h)

    else:
        # Dirichlet: standard tridiagonal (rows kept as is).
        pass

    return A.tocsr()


def generate_laplacian(shape, deltas=None, bcs=None, analytic_normalize=False):
    """
    Build separable FD Laplacian and optionally apply analytic normalization ().

    Parameters
    ----------
    shape : tuple of ints
        Number of grid points per axis (Nx, Ny, ...).
    deltas : tuple of floats or None
        Grid spacings per axis (hx, hy, ...). If None, uses 1.0 for each axis.
    bcs : str or tuple-of-str or None
        Boundary conditions per axis. If None, uses 'dirichlet' on all axes.
    analytic_normalize : bool
        If True, scale the assembled Laplacian by lambda_max = 4 * sum_i (1/h_i^2).
        Returns (A_scaled, lambda_max). If False, returns A only. (We use the same analytical scaling factor as in the Sturm et al. 2025 paper.)

    Returns
    -------
    A or A_scaled
    """

    shape = tuple(shape)
    D = len(shape)
    if deltas is None:
        deltas = tuple([1.0] * D)
    else:
        deltas = tuple(deltas)

    if bcs is None:
        bcs = tuple(["dirichlet"] * D)
    elif isinstance(bcs, str):
        bcs = tuple([bcs] * D)
    else:
        bcs = tuple(bcs)

    # Build 1D operators
    ops_1d = [lap1d_fd(n, h, bc) for (n, h, bc) in zip(shape, deltas, bcs)]

    # Build Kronecker-sum: sum_k (I ⊗ ... ⊗ K_k ⊗ ... ⊗ I) # just like in the third reference paper (Sturm et al. 2015)
    total = None
    for axis, K in enumerate(ops_1d):
        # left identity: product of identity matrices for axes > axis
        kron_left = sp.eye(1, format="csr")
        for j in range(D - 1, axis, -1):
            kron_left = sp.kron(sp.eye(shape[j], format="csr"), kron_left, format="csr")

        # right identity: product for axes < axis
        kron_right = sp.eye(1, format="csr")
        for j in range(0, axis):
            kron_right = sp.kron(
                kron_right, sp.eye(shape[j], format="csr"), format="csr"
            )

        term = sp.kron(kron_left, sp.kron(K, kron_right, format="csr"), format="csr")
        total = term if total is None else (total + term)

    A = total.tocsr()

    if (
        analytic_normalize
    ):  # again the same scaling factor as in the Sturm et al. 2025 paper
        # analytic lambda_max = 4 * sum_i (1 / h_i^2)
        lambda_max = 4.0 * sum((1.0 / (h * h)) for h in deltas)
        if lambda_max <= 0:
            raise ValueError("Computed lambda_max <= 0; check grid spacings.")
        A_scaled = A / lambda_max
        return A_scaled.tocsr()
    else:
        return A


def prepare_v_vector(nqs, v, deltas=None):
    r"""Prepares the normalized vector of function values used in the block encoding.

    Args:
        nqs (list[int]): Number of qubits for each dimension.
        v (Callable[..., float]): Function to block encode.
        deltas (list[float]): Grid spacings for each dimensions.

    Returns:
        ndarray(float): Function values reshaped into a vector ready to be used on the block encoding.
    """
    if deltas is None:
        deltas = [2**-nq for nq in nqs]

    points = [np.arange(2 ** nqs[i]) * deltas[i] for i in range(len(nqs) - 1, -1, -1)]

    axes = np.meshgrid(*points, indexing="ij")
    vs = v(*axes)
    vs_flat = vs.flatten()

    norm = np.linalg.norm(vs_flat)

    return vs_flat / norm


def get_circuit_unitary(qc, nqs, subspace=True):
    r"""Build the matrix representation of a quantum circuit block encoding a Laplacian.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit that block encodes a Laplacian.
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.

    Returns:
        ndarray(float): Representation of the block encoded Laplacian matrix.
    """
    simulator = AerSimulator(method="unitary")
    qc = transpile(qc, simulator, optimization_level=0)

    result = simulator.run(qc).result()
    unitary = result.get_unitary(qc).data.real

    if subspace:
        unitary_subspace = unitary[: 2 ** sum(nqs), : 2 ** sum(nqs)]

        return unitary_subspace

    return unitary


def plot_heatmap(qc, nqs, deltas=None, bcs=None, ncs=101, vmax=1.0):
    r"""Plot the matrix elements for a Laplacian operator.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit that block encodes a Laplacian.
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.
        deltas (list[float]): Grid spacings for each dimension.
        bcs (list[str]): Boundary conditions of the Laplacian.
        ncs (int): Number of colors to use in the plot.
        vmax (float): Color scaling set to -vmax to vmax.
    """
    unitary = get_circuit_unitary(qc, nqs)
    cmap = plt.get_cmap("seismic", ncs)

    plt.imshow(unitary, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.show()
    plt.show()
