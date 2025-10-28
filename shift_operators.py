from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate


class ShiftUp(Gate):
    def __init__(self, num_qubits):
        super().__init__("S+", num_qubits, [])

    def _define(self):
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits - 1, -1, -1):
            if i == 0:
                qc.x(0)

            else:
                cNx = XGate().control(i)
                qc.append(cNx, list(range(i + 1)))

        self.definition = qc


class ShiftDown(Gate):
    def __init__(self, num_qubits):
        super().__init__("S-", num_qubits, [])

    def _define(self):
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            if i == 0:
                qc.x(0)

            else:
                cNx = XGate().control(i)
                qc.append(cNx, list(range(i + 1)))

        self.definition = qc
