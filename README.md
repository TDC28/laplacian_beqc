# Block Encoding Quantum circuits for Laplacian Operators with Mixed Boundary Conditions

Introduction

### Overview

* ***block_encoding.py*** contains the construction Laplacian block encoding quantum circuits. We make use of the Qiskit ecosystem to build, transpile, and simulate the circuits.

* ***demo.ipynb*** is a Jupyter notebook showcasing all our matrix constructions along with correctness checks. The corresponding block encoding quantum circuit for each matrix are also presented.

* ***compare.ipynb*** is a Jupyter notebook comparing our results with state of the art block encoding methods [[1 - 5]](#references) in total gate count, 2-qubit gate count, and block encoding success probability.

* ***utils.py*** contains a few utility functions to help make the code more readable and organized.


### Acknowledgements

This work was carried out as part of the Qiskit Advocate Mentorship Program 2025, organized within
the IBM Qiskit Advocate Program. The authors thank the organizers for facilitating the program and
supporting collaboration within the Qiskit community.

### References

[1] A. Sturm and N. Schillo, “Efficient and explicit block encoding of finite difference discretizations of the laplacian,” (2025),
[arXiv:2509.02429 [quant-ph]](https://arxiv.org/abs/2509.02429).

[2] D. Camps and R. Van Beeumen, "FABLE: Fast Approximate Quantum Circuits for Block-Encodings," (2022), [arXiv:2205.00081 [quant-ph]](https://arxiv.org/abs/2205.00081).

[3] D. Camps, L. Lin, R. V. Beeumen, and C. Yang, “Explicit Quantum Circuits for Block Encodings of Certain Sparse
Matrices,” (2023), [arXiv:2203.10236 [quant-ph]](https://arxiv.org/abs/2203.10236).

[4] C. Sünderhauf, E. Campbell, and J. Camps,  “Block-encoding structured matrices for
data input in quantum computing,” [Quantum 8, 1226 (2024)](https://doi.org/10.22331/q-2024-01-11-1226). 

[5] Z. Li, X. -M. Zhang, C. Yang and G. Zhang, "Binary Tree Block Encoding of Classical Matrix," [IEEE Transactions on Quantum Engineering 7, 1-18 (2026)](https://doi.org/10.1109/TQE.2025.3624699).



