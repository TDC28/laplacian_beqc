import math
from math import ceil, log2

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator


def unique_val_lap(laplacian, Nx):
    """Extract unique diagonal values from a 2D Laplacian matrix."""
    unique_values = [
        laplacian.diagonal(0)[0],  # Main diagonal (A0)
        laplacian.diagonal(1)[0],  # Next lower diagonal (A1)
        laplacian.diagonal(Nx)[0],  # Final diagonal (A2)
    ]
    return unique_values


def make_incrementer(n: int):
    circ = QuantumCircuit(n, name=f"INC{n}")
    for i in range(n - 1, 0, -1):
        circ.mcx(list(range(i)), i)
    circ.x(0)
    return circ.to_gate()


def make_decrementer(n: int):
    inc = make_incrementer(n)
    dec = inc.inverse()
    dec.name = f"DEC{n}"
    return dec


def prep_lap(reg: QuantumRegister, vec: list[float]):
    qc = QuantumCircuit(reg)
    prep = StatePreparation(vec)
    qc.append(prep, reg)
    return qc


def unprep_lap(reg: QuantumRegister, vec: list[float]):
    prep_qc = prep_lap(reg, vec)
    unitary = Operator(prep_qc).data
    inv_unitary = np.conj(unitary.T)
    qc = QuantumCircuit(reg)
    prep_inv_gate = UnitaryGate(inv_unitary)
    qc.append(prep_inv_gate, reg)
    return qc


def oracle_org(qc, dl_reg, j_reg, s_reg, j1, j2, jx, jy):
    oracle_100 = XGate().control(3, ctrl_state="001")
    qc.append(oracle_100, [s_reg[0], s_reg[1], s_reg[2], dl_reg[0]])
    oracle_01_jy = XGate().control(j1 + 1, ctrl_state=("1" + "0" * j1))
    qc.append(oracle_01_jy, jx + [s_reg[1], dl_reg[0]])
    oracle_01_jx = XGate().control(j2 + 1, ctrl_state=("1" + "0" * j2))
    qc.append(oracle_01_jx, jy + [s_reg[2], dl_reg[0]])


def oracle_d(qc: QuantumCircuit, TM: list[float], data_regs: list, s_reg: list):
    """
    Apply a multi‐controlled RX on each data qubit in data_regs,
    with angles 2*arccos(TM[i]) controlled on the bitstring i of s_reg[1:].

    Args:
        qc        : your QuantumCircuit
        TM        : list of length 2**n_controls of amplitudes
        data_regs : list of target qubits (e.g. [data_reg[0]])
        s_reg     : full s register (we’ll use s_reg[1:] as controls)
    """
    controls = [s_reg[1], s_reg[2]]
    n_ctrl = len(controls)

    if len(TM) != 2**n_ctrl:
        raise ValueError(f"TM must have length 2**{n_ctrl} (got {len(TM)})")

    for i, amp in enumerate(TM):
        angle = 2 * np.arccos(amp)
        ctrl_state = format(i, "02b")
        gate = RXGate(angle).control(2, ctrl_state=ctrl_state)
        qc.append(gate, controls + [data_regs[0]])

    return qc


def sunderhauff_block_encoding(
    Nx, Ny, scaled_laplacian, v_normed=None, save_unitary=True
):

    unique_values_laplacian = unique_val_lap(scaled_laplacian, Nx)
    unique_values_laplacian.append(0)

    data_reg = QuantumRegister(1, "data")
    dlt_reg = QuantumRegister(1, "dlt")
    s_reg = QuantumRegister(3, "s")  # Corresponds to s0, s1, s2
    j1, j2 = int(math.log2(Nx)), int(math.log2(Ny))
    j_reg = QuantumRegister(j1 + j2, "j")
    jx, jy = j_reg[:j1], j_reg[j1:]
    qc = QuantumCircuit(j_reg, s_reg, dlt_reg, data_reg)

    if v_normed is not None:
        qc.append(StatePreparation(v_normed), j_reg)

    vec = [
        np.sqrt(1 / 5),
        0,
        np.sqrt(1 / 5),
        np.sqrt(1 / 5),
        np.sqrt(1 / 5),
        np.sqrt(1 / 5),
        0,
        0,
    ]

    prep = prep_lap(s_reg, vec)
    qc.append(prep, s_reg)

    add_1_to_jy = make_incrementer(j2).control(2, ctrl_state="10")
    add_1_to_j = make_incrementer(int(j1 + j2)).control(2, ctrl_state="10")
    qc.append(add_1_to_jy, [s_reg[0], s_reg[2]] + jy)  # [::-1]
    qc.append(add_1_to_j, [s_reg[0], s_reg[1]] + list(j_reg))  # [::-1]

    oracle_org(qc, dlt_reg, j_reg, s_reg, j1, j2, jx, jy)
    qc.z(data_reg[0])
    oracle_d(qc, unique_values_laplacian, data_reg, s_reg)
    qc.cx(s_reg[1], s_reg[0])
    qc.cx(s_reg[2], s_reg[0])

    sub_1_to_jx = make_decrementer(int(j1 + j2)).control(2, ctrl_state="10")
    qc.append(sub_1_to_jx, [s_reg[0], s_reg[1]] + list(j_reg))
    sub_1_to_j = make_decrementer(j2).control(2, ctrl_state="10")
    qc.append(sub_1_to_j, [s_reg[0], s_reg[2]] + jy)

    prep_dagger = unprep_lap(s_reg, vec)
    qc.append(prep_dagger, s_reg)

    if save_unitary:
        qc.save_unitary()

    return qc
