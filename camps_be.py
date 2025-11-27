import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

from shift_operators import ShiftDown, ShiftUp


def equal_superposition(k, nq=None):
    """Creates a quantum circuit that creates an equal superposition between |0> and |k-1>.

    Args:
        k: Number of states in superposition.
        nq: Number of qubits in the register.

    Returns:
        a qiskit.QuantumCircuit object containing a circuit that maps a state 0 to equal superposition state.
    """
    if nq is None:
        nq = int(np.ceil(np.log2(k)))

    qc = QuantumCircuit(nq)

    if k == 1:
        return qc

    elif (k & (k - 1)) == 0:
        qc.h(range(int(np.log2(k))))

        return qc

    n_extra_states = k - 2 ** (nq - 1)
    theta = 2 * np.arcsin(np.sqrt(n_extra_states / k))

    qc.ry(theta, nq - 1)
    qc.x(nq - 1)
    qc.ch(nq - 1, range(nq - 1))
    qc.x(nq - 1)

    circuit = equal_superposition(n_extra_states, nq - 1)
    gate = circuit.to_gate().control(1)
    qc.append(gate, [nq - 1] + list(range(nq - 1)))

    return qc


def camps_block_encoding(nq, bcs):
    r"""Build the quantum circuit for the block encoding of an N-dimensional Laplacian operator

    Args:
        nq (int): Number of qubits per dimensions. Corresponds to 2**nq grid points per dimension.
        bcs (list[str]): Boundary conditions of the laplacian. Each item in the list is either "periodic"
            or "dirichlet". The length of the list determines the number of dimensions.

    Returns:
        a numpy.ndarray representing the unitary operation equivalent to the quantum circuit to block encode
        the Laplacian operator.
    """
    if len(list(filter(lambda x: x != "dirichlet" and x != "periodic", bcs))) != 0:
        raise ValueError("Invalid boundary conditions")

    d = len(bcs)
    k = int(np.ceil(np.log2(d)))

    theta_diag = 2 * np.arccos(1 / 2 - 1)
    theta_off = 2 * np.arccos(-1 / 4)

    l_reg = QuantumRegister(2, "l")
    data_reg = QuantumRegister(1, "data")
    j_regs = []

    simulator = AerSimulator(method="unitary")

    for i in range(d):
        j_regs.append(QuantumRegister(nq, f"j^{{({i})}}"))

    ry_diag = RYGate(theta_diag)
    ry_off = RYGate(theta_off)
    shift_up = ShiftUp(nq)
    shift_down = ShiftDown(nq)

    if k == 0:
        qc = QuantumCircuit(*j_regs, data_reg, l_reg)

        # Defining c-gates
        ry_diag_mc0 = ry_diag.control(2, ctrl_state="00")
        ry_off_mc1 = ry_off.control(2, ctrl_state="01")
        ry_off_mc2 = ry_off.control(2, ctrl_state="10")
        su_mc1 = shift_up.control(2, ctrl_state="01")
        sd_mc2 = shift_down.control(2, ctrl_state="10")

        # Building circuit
        qc.h(l_reg)
        qc.append(ry_diag_mc0, l_reg[:] + data_reg[:])
        qc.append(ry_off_mc1, l_reg[:] + data_reg[:])
        qc.append(ry_off_mc2, l_reg[:] + data_reg[:])

        if bcs[0] == "dirichlet":
            ry_dirichlet = RYGate(np.pi - theta_off)

            ryd_mc1 = ry_dirichlet.control(nq + 2, ctrl_state="1" * nq + "01")
            ryd_mc2 = ry_dirichlet.control(nq + 2, ctrl_state="0" * nq + "10")

            qc.append(ryd_mc1, l_reg[:] + j_regs[0][:] + data_reg[:])
            qc.append(ryd_mc2, l_reg[:] + j_regs[0][:] + data_reg[:])

        qc.append(sd_mc2, l_reg[:] + j_regs[0][:])
        qc.append(su_mc1, l_reg[:] + j_regs[0][:])
        qc.h(l_reg)
        qc.save_unitary()

    else:
        k_reg = QuantumRegister(k, "k")
        qc = QuantumCircuit(*j_regs, data_reg, l_reg, k_reg)
        equal_superposition_circuit = equal_superposition(d)
        inv_equal_superposition_circuit = equal_superposition_circuit.inverse()

        qc.h(l_reg)
        qc.append(equal_superposition_circuit.to_instruction(), k_reg)
        qc.barrier()

        for i in range(len(bcs)):
            # Control bitstring for k register
            k_ctrl = bin(i)[2:].zfill(k)

            # Defining c-gates
            ry_diag_mc0 = ry_diag.control(k + 2, ctrl_state="00" + k_ctrl)
            ry_off_mc1 = ry_off.control(k + 2, ctrl_state="01" + k_ctrl)
            ry_off_mc2 = ry_off.control(k + 2, ctrl_state="10" + k_ctrl)
            su_mc1 = shift_up.control(k + 2, ctrl_state="01" + k_ctrl)
            sd_mc2 = shift_down.control(k + 2, ctrl_state="10" + k_ctrl)

            # Building circuit
            qc.append(ry_diag_mc0, k_reg[:] + l_reg[:] + data_reg[:])
            qc.append(ry_off_mc1, k_reg[:] + l_reg[:] + data_reg[:])
            qc.append(ry_off_mc2, k_reg[:] + l_reg[:] + data_reg[:])

            if bcs[i] == "dirichlet":
                ry_dirichlet = RYGate(np.pi - theta_off)
                ryd_mc1 = ry_dirichlet.control(
                    k + nq + 2, ctrl_state="1" * nq + "01" + k_ctrl
                )
                ryd_mc2 = ry_dirichlet.control(
                    k + nq + 2, ctrl_state="0" * nq + "10" + k_ctrl
                )

                qc.append(ryd_mc1, k_reg[:] + l_reg[:] + j_regs[i][:] + data_reg[:])
                qc.append(ryd_mc2, k_reg[:] + l_reg[:] + j_regs[i][:] + data_reg[:])

            qc.append(sd_mc2, k_reg[:] + l_reg[:] + j_regs[i][:])
            qc.append(su_mc1, k_reg[:] + l_reg[:] + j_regs[i][:])
            qc.barrier()

        qc.h(l_reg)
        qc.append(inv_equal_superposition_circuit.to_instruction(), k_reg)
        qc.save_unitary()

    return qc
