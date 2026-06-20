import os
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_laplacian


# ==========================================================
# Utility function to plot matrices
# ==========================================================

def save_matrix_plot(A, title, filename):
    """
    Save a heatmap visualization of a matrix.
    """

    A_dense = A.toarray()

    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("seismic", 101)
    im = plt.imshow(A_dense, cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.colorbar(im)

    plt.title(title)
    plt.xlabel("Column index")
    plt.ylabel("Row index")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ==========================================================
# Create output directories
# ==========================================================

os.makedirs("test_outputs/1D", exist_ok=True)
os.makedirs("test_outputs/2D", exist_ok=True)


# ==========================================================
# 1D tests
# ==========================================================

N = 16

bc_tests_1d = {
    "dirichlet": {
        "bcs": "dirichlet",
        "robin_coeffs": None,
    },

    "neumann": {
        "bcs": "neumann",
        "robin_coeffs": None,
    },

    "periodic": {
        "bcs": "periodic",
        "robin_coeffs": None,
    },

    "robin": {
        "bcs": "robin",
        # alpha0, beta0, alphaL, betaL
        "robin_coeffs": (
            (1.0, 2.0, 1.0, 4.0),
        ),
    },
}


for name, params in bc_tests_1d.items():

    A = generate_laplacian(
        shape=(N,),
        bcs=params["bcs"],
        robin_coeffs=params["robin_coeffs"],
        analytic_normalize=True
    )

    print(
    f"{name}: min={A.min():.3f}, max={A.max():.3f}"
)

    print(
        f"1D {name:10s}: "
        f"shape={A.shape}, "
        f"nnz={A.nnz}, "
        f"symmetric={np.allclose(A.toarray(), A.toarray().T)}"
    )

    save_matrix_plot(
        A,
        f"1D Laplacian - {name}",
        f"test_outputs/1D/{name}.png",
    )


# ==========================================================
# 2D tests
# ==========================================================

Nx, Ny = 8, 8


bc_tests_2d = {
    "dirichlet": {
        "bcs": ("dirichlet", "dirichlet"),
        "robin_coeffs": None,
    },

    "neumann": {
        "bcs": ("neumann", "neumann"),
        "robin_coeffs": None,
    },

    "periodic": {
        "bcs": ("periodic", "periodic"),
        "robin_coeffs": None,
    },

    "robin": {
        "bcs": ("robin", "robin"),

        # x-direction Robin, y-direction Robin
        "robin_coeffs": (
            (1.0, 2.0, 1.0, 4.0),
            (0.5, 1.0, 1.5, 3.0),
        ),
    },
}


for name, params in bc_tests_2d.items():

    A = generate_laplacian(
        shape=(Nx, Ny),
        bcs=params["bcs"],
        robin_coeffs=params["robin_coeffs"],
        analytic_normalize=True
    )

    print(
    f"{name}: min={A.min():.3f}, max={A.max():.3f}"
)

    print(
        f"2D {name:10s}: "
        f"shape={A.shape}, "
        f"nnz={A.nnz}, "
        f"symmetric={np.allclose(A.toarray(), A.toarray().T)}"
    )

    save_matrix_plot(
        A,
        f"2D Laplacian - {name}",
        f"test_outputs/2D/{name}.png",
    )

print("Plots saved in test_outputs/")