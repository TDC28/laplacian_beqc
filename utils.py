import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from qiskit import transpile
from qiskit_aer import AerSimulator


def lap1d_fd(n, h=1.0, bc="dirichlet", robin_coeffs=None):
    r"""Build 1D second-order finite difference Laplacian matrix.

    Args:
        n (int):
            Number of grid points.
        h (float):
            Grid spacing.
        bc (str):
            Boundary condition type:
            'dirichlet', 'neumann', 'periodic', or 'robin'.
        robin_coeffs (tuple[float, float, float, float] | None):
            Robin coefficients (alpha0, beta0, alphaL, betaL)
            for the left and right boundaries:
                alpha*u + beta*du/dx = 0

    Returns:
        scipy.sparse.csr_matrix:
            1D finite-difference Laplacian scaled by 1/h^2.
    """

    if n < 1:
        raise ValueError("n must be >= 1")

    main = -2.0 / (h * h)
    off = 1.0 / (h * h)

    A = sp.diags(
        [
            off * np.ones(n - 1),
            main * np.ones(n),
            off * np.ones(n - 1),
        ],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="lil",
    )

    if bc == "dirichlet":
        pass

    elif bc == "periodic":
        if n == 1:
            A[0, 0] = 0.0
        else:
            A[0, -1] = off
            A[-1, 0] = off

    elif bc == "neumann":
        if n == 1:
            A[0, 0] = 0.0
        else:
            A[0, 0] = -off
            A[0, 1] = off
            A[-1, -2] = off
            A[-1, -1] = -off

    elif bc == "robin":
        if robin_coeffs is None:
            raise ValueError(
                "robin_coeffs must be provided when bc='robin'."
            )

        if len(robin_coeffs) != 4:
            raise ValueError(
                "robin_coeffs must be "
                "(alpha0, beta0, alphaL, betaL)."
            )

        alpha0, beta0, alphaL, betaL = robin_coeffs

        if beta0 == 0 or betaL == 0:
            raise ValueError(
                "beta values cannot be zero for Robin BC. "
                "Use Dirichlet BC instead."
            )

        a0 = alpha0 / beta0
        aL = alphaL / betaL

        if n == 1:
            A[0, 0] = (a0 * h - 1.0) / (h * h)
        else:
            A[0, 0] = (a0 * h - 1.0) / (h * h)
            A[0, 1] = off

            A[-1, -2] = off
            A[-1, -1] = (aL * h - 1.0) / (h * h)

        if abs(a0*h - 1) > 4 or abs(aL*h - 1) > 4:
            warnings.warn(
                "Robin coefficients may violate analytic normalization bounds."
            )


    else:
        raise ValueError(
            "bc must be 'dirichlet', 'neumann', "
            "'periodic', or 'robin'."
        )

    return A.tocsr()


def generate_laplacian(
    shape,
    deltas=None,
    bcs=None,
    analytic_normalize=False,
    robin_coeffs=None,
):
    r"""Build an N-dimensional finite difference Laplacian matrix.

    Args:
        shape (tuple[int, ...]): Number of grid points per axis.
        deltas (tuple[float, ...]): Grid spacings per axis (hx, hy, ...). If None, defaults to 1.0 for each axis.
        bcs: (tuple[str | tuple[float, float, float, float]], ...] | str | None): Boundary conditions per axis. If None, uses 'dirichlet' on all axes.
        analytic_normalize (bool): If True, scale the assembled Laplacian by 4 * sum_i (1/h_i^2).

    Returns:
        scipy.sparse.csr_matrix: Sparse N-dimensional finite difference Laplacian matrix.
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

    # Process Robin coefficients per axis
    if robin_coeffs is None:
        robin_coeffs = tuple([None] * D)

    else:
        robin_coeffs = tuple(robin_coeffs)

    if len(robin_coeffs) != D:
        raise ValueError(
            "robin_coeffs must have one entry per dimension."
        )

    # Build 1D operators
    ops_1d = [
    lap1d_fd(n, h, bc, robin)
    for (n, h, bc, robin) in zip(
        shape,
        deltas,
        bcs,
        robin_coeffs,
    )
]

    # Build Kronecker-sum: sum_k (I ⊗ ... ⊗ K_k ⊗ ... ⊗ I) 
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
        numpy.ndarray: Function values reshaped into a vector ready to be used on the block encoding.
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
    r"""Build the matrix representation of a Laplacian block encoding quantum circuit.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit that block encodes a Laplacian.
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.

    Returns:
        numpy.ndarray: Representation of the block encoded Laplacian matrix.
    """
    print("Function call - get_circuit_unitary")
    simulator = AerSimulator(method="unitary")
    print("Simulator loaded")
    print("Starting transpilation")
    qc_transpiled = transpile(qc, simulator)   # This line causes a crash when called with StatePreparation
    print("transpiled")

    result = simulator.run(qc_transpiled).result()
    unitary = result.get_unitary(qc_transpiled).data.real

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
