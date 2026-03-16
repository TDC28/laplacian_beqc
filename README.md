[![arXiv](https://img.shields.io/badge/arXiv-2603.12405-b31b1b.svg)](https://arxiv.org/abs/2603.12405)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum%20Computing-6929C4.svg)](https://qiskit.org/)
[![GitHub last commit](https://img.shields.io/github/last-commit/TDC28/laplacian_beqc)](https://github.com/TDC28/laplacian_beqc)

# Explicit Block Encodings of Discrete Laplacians with Mixed Boundary Conditions

This repository contains the reference implementation accompanying the paper:

**Explicit Block Encodings of Discrete Laplacians with Mixed Boundary Conditions**  
Alexandre Boutot and Viraj Dsouza  
[arXiv:2603.12405](https://arxiv.org/abs/2603.12405) (2026)

The code provides implementations of quantum circuits for block encoding discrete Laplacian operators with Dirichlet, periodic, and Neumann boundary conditions in arbitrary spatial dimensions.

---

# Overview

The repository contains the following components:

### `laplacian_beqc.py`
Implementation of the block encoding circuits for discrete Laplacians.  
The circuits are constructed using the **Qiskit** ecosystem and can be transpiled and simulated using standard Qiskit tools.

### `demo.ipynb`
A Jupyter notebook demonstrating the construction of the discrete Laplacian matrices and the corresponding block encoding quantum circuits.  
The notebook includes correctness checks verifying that the encoded unitary reproduces the intended Laplacian operator.

### `comparison.ipynb`
A benchmarking notebook comparing the proposed construction with several recent block encoding methods [[1 - 5]](#references). The comparisons include:

- total gate count  
- two-qubit gate count  
- block encoding success probability  

### `utils.py`
Utility functions used throughout the repository.


The `figs` folder contains the plots used in the paper. The `bitble_circuits` folder contains QASM files for BITBLE block-encoding circuits [[5]](#references), which are used for comparison.

---

# Requirements

The implementation relies primarily on the **Qiskit** framework.

Install dependencies with:

```bash
pip install qiskit numpy matplotlib qiskit-aer scipy
```

To use the FABLE implementation [[2]](#references):

```bash
pip install fable-circuits
```
---

# Citation

If you use this code or build upon this work, please cite the associated paper:

```bibtex
@misc{boutot2026explicitblockencodingsdiscrete,
      title={Explicit Block Encodings of Discrete Laplacians with Mixed Boundary Conditions},
      author={Alexandre Boutot and Viraj Dsouza},
      year={2026},
      eprint={2603.12405},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.12405}
}

```
---

# License 

This project is released as open source under the MIT License.
You are free to use, modify, and distribute this code for research and educational purposes.
If this repository contributes to your work, please cite the associated paper.

See the [LICENSE file](LICENSE.txt) for full details.

---

### Acknowledgements

This work was carried out as part of the Qiskit Advocate Mentorship Program 2025, organized within
the IBM Qiskit Advocate Program. The authors thank the organizers for facilitating the program and
supporting collaboration within the Qiskit community.

---

### References

[1] A. Sturm and N. Schillo, “Efficient and explicit block encoding of finite difference discretizations of the laplacian,” (2025),
[arXiv:2509.02429 [quant-ph]](https://arxiv.org/abs/2509.02429).

[2] D. Camps and R. Van Beeumen, "FABLE: Fast Approximate Quantum Circuits for Block-Encodings," (2022), [arXiv:2205.00081 [quant-ph]](https://arxiv.org/abs/2205.00081).

[3] D. Camps, L. Lin, R. V. Beeumen, and C. Yang, “Explicit Quantum Circuits for Block Encodings of Certain Sparse
Matrices,” (2023), [arXiv:2203.10236 [quant-ph]](https://arxiv.org/abs/2203.10236).

[4] C. Sünderhauf, E. Campbell, and J. Camps,  “Block-encoding structured matrices for
data input in quantum computing,” [Quantum 8, 1226 (2024)](https://doi.org/10.22331/q-2024-01-11-1226). 

[5] Z. Li, X. -M. Zhang, C. Yang and G. Zhang, "Binary Tree Block Encoding of Classical Matrix," [IEEE Transactions on Quantum Engineering 7, 1-18 (2026)](https://doi.org/10.1109/TQE.2025.3624699).
