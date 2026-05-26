import numpy as np
from numba import jit
from scipy.stats import unitary_group


class QuantumGate:
    """Base class for quantum gates"""

    def __init__(self, qubits, matrix):
        """
        Args:
            qubits: Tuple of qubit indices (e.g., (0,1) for 2-qubit gate)
            matrix: Unitary matrix (2^n × 2^n for n-qubit gate)
        """
        self.qubits = tuple(qubits)  # no-sort
        self.matrix = np.array(matrix, dtype=np.complex128)
        self.n_qubits = len(qubits)

        # Validate unitary
        dim = 2**self.n_qubits
        assert self.matrix.shape == (dim, dim), "Matrix dimension mismatch"

    def __repr__(self):
        return f"{self.__class__.__name__}(qubits={self.qubits})"


class SingleQubitGate(QuantumGate):
    """Single-qubit gate base class"""

    def __init__(self, qubit, matrix):
        super().__init__((qubit,), matrix)


class TwoQubitGate(QuantumGate):
    """Two-qubit gate base class"""

    def __init__(self, qubit1, qubit2, matrix):
        super().__init__((qubit1, qubit2), matrix)


# Standard gates
class HGate(SingleQubitGate):
    """Hadamard gate"""

    def __init__(self, qubit):
        matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        super().__init__(qubit, matrix)


class XGate(SingleQubitGate):
    """Pauli X (NOT) gate"""

    def __init__(self, qubit):
        matrix = np.array([[0, 1], [1, 0]])
        super().__init__(qubit, matrix)


class YGate(SingleQubitGate):
    """Pauli Y gate"""

    def __init__(self, qubit):
        matrix = np.array([[0, -1j], [1j, 0]])
        super().__init__(qubit, matrix)


class ZGate(SingleQubitGate):
    """Pauli Z gate"""

    def __init__(self, qubit):
        matrix = np.array([[1, 0], [0, -1]])
        super().__init__(qubit, matrix)


class SGate(SingleQubitGate):
    """S (Phase) gate"""

    def __init__(self, qubit):
        matrix = np.array([[1, 0], [0, 1j]])
        super().__init__(qubit, matrix)


class TGate(SingleQubitGate):
    """T gate"""

    def __init__(self, qubit):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        super().__init__(qubit, matrix)


class CNOTGate(TwoQubitGate):
    """CNOT gate"""

    def __init__(self, control, target):
        matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        super().__init__(control, target, matrix)


class CZGate(TwoQubitGate):
    """Controlled-Z gate"""

    def __init__(self, qubit1, qubit2):
        matrix = np.diag([1, 1, 1, -1])
        super().__init__(qubit1, qubit2, matrix)


class RXGate(SingleQubitGate):
    """Rotation around X axis: exp(i theta X)"""

    def __init__(self, qubit, theta):
        c, s = np.cos(theta), np.sin(theta)
        matrix = np.array([[c, 1j * s], [1j * s, c]], dtype=np.complex128)
        super().__init__(qubit, matrix)


class RZZGate(TwoQubitGate):
    """exp(i theta Z⊗Z) diagonal gate"""

    def __init__(self, qubit1, qubit2, theta):
        p, m = np.exp(1j * theta), np.exp(-1j * theta)
        matrix = np.diag([p, m, m, p]).astype(np.complex128)
        super().__init__(qubit1, qubit2, matrix)


class RZGate(SingleQubitGate):
    """Single-qubit rotation around Z axis."""

    def __init__(self, qubit, theta):
        p, m = np.exp(-1j * theta / 2), np.exp(1j * theta / 2)
        matrix = np.array([[p, 0], [0, m]], dtype=np.complex128)
        super().__init__(qubit, matrix)


class RYGate(SingleQubitGate):
    """Single-qubit rotation around Y axis."""

    def __init__(self, qubit, theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        matrix = np.array([[c, -s], [s, c]], dtype=np.complex128)
        super().__init__(qubit, matrix)


class RandomTwoQubitGate(TwoQubitGate):
    """Random Haar-uniform 2-qubit gate"""

    def __init__(self, qubit1, qubit2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        matrix = unitary_group.rvs(4)
        super().__init__(qubit1, qubit2, matrix)


def haar_random_unitary(dimension, seed=None):
    """
    Generate Haar-random unitary matrix

    Args:
        dimension: Matrix dimension (2^n for n qubits)
        seed: Random seed
    Returns:
        Unitary matrix
    """
    if seed is not None:
        np.random.seed(seed)
    return unitary_group.rvs(dimension)
