# Block Encoding Quantum circuits for Laplacian Operators with Mixed Boundary Conditions

As our submission for the Qiskit Advocate Mentorship Program (QAMP) 2025, we present novel block encoding quantum circuits for Laplacian operators. This repository presents our circuit constructions and compares them with other approaches to block encoding.

### Overview

* ***block_encoding.py*** contains the code to build our Laplacian block encoding quantum circuits. We make use of the **Qiskit** ecosystem to build, transpile, and simulate the circuits.

* ***demo.ipynb*** is a Jupyter notebook showcasing all our matrix constructions along with correctness checks. The corresponding block encoding quantum circuit for each matrix are also presented.

* ***compare.ipynb*** is a Jupyter notebook comparing our results with state of the art block encoding methods [[1 - 4]](#references) in total gate count, 2-qubit gate count, and block encoding success probability.

* ***utils.py*** contains a few utility functions to help make the code more readable and organized.


### References

[1] A. Sturm and N. Schillo, “Efficient and explicit block encoding of finite difference discretizations of the laplacian,” (2025),
[arXiv:2509.02429 [quant-ph]](https://arxiv.org/abs/2509.02429).

[2] D. Camps and R. Van Beeumen, "FABLE: Fast Approximate Quantum Circuits for Block-Encodings," (2022), [arXiv:2205.00081 [quant-ph]](https://arxiv.org/abs/2205.00081).

[3] D. Camps, L. Lin, R. V. Beeumen, and C. Yang, “Explicit Quantum Circuits for Block Encodings of Certain Sparse
Matrices,” (2023), [arXiv:2203.10236 [quant-ph]](https://arxiv.org/abs/2203.10236).

[4] C. Sünderhauf, E. Campbell, and J. Camps, Quantum 8, 1226 (2024).
