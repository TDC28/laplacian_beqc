import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, XGate
from qiskit_aer import AerSimulator

from shift_operators import ShiftDown, ShiftUp


def prepare_k_register(deltas):
    r"""Prepares k register into state Σ sqrt(omega_i) |i> where omega_i is inversely proportional to the square
        of the grid spacing of the i'th dimension.

    Args:
        deltas (list[float]): The grid spacings for each dimension.

    Returns:
        qiskit.circuit.Instruction: An instruction that performs the desired state preparation.
    """
    k = int(np.ceil(np.log2(len(deltas))))
    omegas_non_normalized = 4 / np.array(deltas) ** 2
    omegas = omegas_non_normalized / np.sum(omegas_non_normalized)
    even = np.sum(omegas[::2])

    qc = QuantumCircuit(k)
    theta0 = 2 * np.arccos(np.sqrt(even))
    qc.ry(theta0, 0)

    if k == 2:
        qc.cry(2 * np.arccos(np.sqrt(omegas[0] / even)), *qc.qubits, ctrl_state=0)
        qc.cry(2 * np.arccos(np.sqrt(omegas[1] / (1 - even))), *qc.qubits)

    elif k > 2:
        raise NotImplementedError(
            f"{len(deltas)}-dimensional Laplacian not implemented."
        )

    instr = qc.to_instruction(label="prep_k")
    instr_inv = instr.inverse()
    instr_inv.label = "prep_k_inv"

    return instr, instr_inv


def generate_laplacian_block_encoding(
    nqs, deltas=None, bcs=None, vs=None, save_unitary=True
):
    r"""Build the quantum circuit for the block encoding of an N-dimensional Laplacian operator

    Args:
        nqs (list[int]): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.
        deltas (list[float]): Grid spacings for each dimension.
        bcs (list[str]): Boundary conditions of the laplacian. Each item in the list is either "periodic"
            or "dirichlet". Defaults to Dirichlet BCs.
        vs (list[float]): The function values at each point. Length should match the total number of grid points.

    Returns:
        qiskit.QuantumCircuit: Circuit that block encodes the desired Laplacian operator.
    """
    if deltas is None:
        deltas = [1.0] * len(nqs)

    if bcs is None:
        bcs = ["dirichlet"] * len(nqs)

    assert (
        len(
            list(
                filter(
                    lambda x: x != "dirichlet" and x != "periodic" and x != "neumann",
                    bcs,
                )
            )
        )
        == 0
    ), "Invalid boundary conditions"

    d = len(nqs)
    k = int(np.ceil(np.log2(d)))

    # Defining registers
    l_reg = QuantumRegister(2, "l")
    del_reg = QuantumRegister(1, "del")
    j_regs = [QuantumRegister(nqs[i], f"j^{{({i})}}") for i in range(d)]
    k_reg = QuantumRegister(k, "k")

    if k == 0:
        qc = QuantumCircuit(*j_regs, del_reg, l_reg)

        if vs is not None:
            all_qubits = [q for reg in j_regs for q in reg]
            qc.append(StatePreparation(vs), all_qubits)

        qc.h(l_reg)
        qc.z(l_reg)

        # Apply dirichlet extra gates if BC is Dirichlet
        if bcs[0] == "dirichlet":
            cx0 = XGate().control(nqs[0] + 2, ctrl_state="0" * (nqs[0] + 2))
            cx1 = XGate().control(nqs[0] + 2, ctrl_state="1" * (nqs[0] + 2))

            qc.append(cx0, j_regs[0][:] + l_reg[:] + del_reg[:])
            qc.append(cx1, j_regs[0][:] + l_reg[:] + del_reg[:])

        elif bcs[0] == "neumann":
            cx0 = XGate().control(nqs[0] + 2, ctrl_state="0" * (nqs[0] + 2))
            cx1 = XGate().control(nqs[0] + 2, ctrl_state="01" + "0" * nqs[0])
            cx2 = XGate().control(nqs[0] + 2, ctrl_state="1" * (nqs[0] + 2))
            cx3 = XGate().control(nqs[0] + 2, ctrl_state="01" + "1" * nqs[0])

            qc.append(cx0, j_regs[0][:] + l_reg[:] + del_reg[:])
            qc.append(cx1, j_regs[0][:] + l_reg[:] + del_reg[:])
            qc.append(cx2, j_regs[0][:] + l_reg[:] + del_reg[:])
            qc.append(cx3, j_regs[0][:] + l_reg[:] + del_reg[:])

        # Apply shift operators
        csu = ShiftUp(nqs[0]).control(1)
        csd = ShiftDown(nqs[0]).control(1, ctrl_state=0)

        qc.append(csd, [l_reg[1]] + j_regs[0][:])
        qc.append(csu, [l_reg[0]] + j_regs[0][:])

        qc.h(l_reg)

        if save_unitary:
            qc.save_unitary()

    else:
        qc = QuantumCircuit(*j_regs, del_reg, l_reg, k_reg)

        if vs is not None:
            all_qubits = [q for reg in j_regs for q in reg]
            qc.append(StatePreparation(vs), all_qubits)

        k_prep, k_prep_inv = prepare_k_register(deltas)
        qc.append(k_prep, k_reg)
        qc.h(l_reg)
        qc.z(l_reg)
        qc.barrier()

        for i in range(d):
            # Control bitstring for k register
            k_ctrl = bin(i)[2:].zfill(k)

            # Apply dirichlet extra gates if current BC is Dirichlet
            if bcs[i] == "dirichlet":
                cx0 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "0" * (nqs[i] + 2)
                )
                cx1 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "1" * (nqs[i] + 2)
                )

                qc.append(cx0, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])
                qc.append(cx1, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])

            if bcs[i] == "neumann":
                cx0 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "0" * (nqs[i] + 2)
                )
                cx1 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "01" + "0" * nqs[i]
                )
                cx2 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "1" * (nqs[i] + 2)
                )
                cx3 = XGate().control(
                    nqs[i] + k + 2, ctrl_state=k_ctrl + "01" + "1" * nqs[i]
                )

                qc.append(cx0, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])
                qc.append(cx1, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])
                qc.append(cx2, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])
                qc.append(cx3, j_regs[i][:] + l_reg[:] + k_reg[:] + del_reg[:])

            # Apply shift operators
            csu = ShiftUp(nqs[i]).control(k + 1, ctrl_state=k_ctrl + "1")
            csd = ShiftDown(nqs[i]).control(k + 1, ctrl_state=k_ctrl + "0")

            qc.append(csd, [l_reg[1]] + k_reg[:] + j_regs[i][:])
            qc.append(csu, [l_reg[0]] + k_reg[:] + j_regs[i][:])
            qc.barrier()

        qc.h(l_reg)
        qc.append(k_prep_inv, k_reg)

        if save_unitary:
            qc.save_unitary()

    return qc
