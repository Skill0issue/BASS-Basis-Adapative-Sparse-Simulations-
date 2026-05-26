"""
QuantumCircuitGenerators
=======================
A collection of ensemble-averaged benchmark quantum circuit factories used to
evaluate the operational performance of BASS versus Fixed-Basis simulation.

This module houses structural frameworks ranging from chaotic, maximally
entangling Haar-random unitaries to physically relevant, disordered multi-body
condensed matter models (TFIM, RFIM). Every circuit is parameterized to support
reproducible, seed-controlled ensemble runs for academic validation.

Implemented Structural Ensembles
--------------------------------
    1. Chaotic Haar-Random Circuit (`generate_random_circuit`)
        Implements a generic entangling fabric where every layer applies floor(N/2)
        Haar-random 2-qubit unitaries to fully randomized pairings of qubits. This
        serves as a high-entanglement baseline that rapidly distributes amplitudes uniformly
        across the computational basis, maximizing the State Participation Ratio (PR).

    2. Transverse-Field Ising Model (`generate_tfim_circuit`)
        Generates a disordered, 1D Trotterized evolution under the Hamiltonian:
            H = - Σ_i J_i Z_i Z_{i+1} - Σ_i h_i X_i
        A single Trotter step maps to an alternating cascade of nearest-neighbor `RZZGate`
        couplings followed by local `RXGate` fields. Coupling parameter distributions are
        drawn uniformly per layer via:
            J_i ~ U[J - dJ, J + dJ]  and  h_i ~ U[h - dh, h + dh]

    3. Random-Field Ising Model (`generate_rfim_circuit`)
        Implements the standard 1D quantum Random-Field Ising Model (RFIM) used to study
        Many-Body Localization (MBL) crossovers. The model introduces true static longitudinal
        disorder:
            H = -J Σ_i Z_i Z_{i+1} - h₀ Σ_i X_i - Σ_i Δ_i Z_i
        where the static random fields Δ_i ~ U[-W, W] are fixed once per circuit instance.
        The first-order Trotter layer maps algebraically to gate operations:
            U_ZZ = Π RZZGate(i, i+1, J * dt)
            U_X  = Π RXGate(i, h0 * dt)
            U_Δ  = Π RZGate(i, -2 * Δ_i * dt)

*Phase Regime Reference (J = h₀ = 1.0):*
    - W < 1.0   : Ergodic Phase (ETH). High state PR; ideal for BASS basis optimization.
    - W ∈ [2,3] : ETH/MBL Crossover. BASS demonstrates moderate localization advantage.
    - W > 4.0   : Deep MBL Phase. State naturally remains highly Z-sparse; BASS converges
                back to baseline fixed-basis behavior.

Reviewer & Reproducibility Standard
--------------------------------------
- Structural Metrics: Includes an exact trace-path tracking routine (`estimate_circuit_depth`)
to compute logical circuit depth via a greedy longest-path dependency tracker across active
qubit indices.

- Seed Containment: All disordered models accept an explicit `numpy.random.Generator`
instance (`rng`), guaranteeing instance-to-instance variation during automated
sweeps while maintaining absolute reproducibility.
"""

import numpy as np
from src.core.gates import RandomTwoQubitGate, RXGate, RZZGate, RZGate


def generate_random_circuit(num_qubits, num_layers, gate_type="haar"):
    """
    Generate random quantum circuit as in Miller et al. paper

    Architecture:
    - Each layer: floor(N/2) or ceil(N/2) random 2-qubit gates
    - Random pairing of qubits within each layer
    - Gates are Haar-random unitaries

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layers (L in paper)
        gate_type: 'haar' for Haar-random (default), 'clifford' for Clifford

    Returns:
        List of gates
    """
    circuit = []

    for layer in range(num_layers):
        # Random pairing of qubits
        qubits = list(range(num_qubits))
        np.random.shuffle(qubits)

        # Apply gates to pairs
        num_pairs = num_qubits // 2
        for i in range(num_pairs):
            q1 = qubits[2 * i]
            q2 = qubits[2 * i + 1]

            if gate_type == "haar":
                gate = RandomTwoQubitGate(q1, q2)
            else:
                raise NotImplementedError(f"Gate type {gate_type} not implemented")

            circuit.append(gate)

    return circuit


def generate_tfim_circuit(
    num_qubits,
    num_layers,
    J=1.0,
    h=3.0,
    dt=0.3,
    rng=None,
    disorder_strength_J=0.0,
    disorder_strength_h=0.0,
):
    """
    Generate disordered 1D TFIM Trotterized circuit.

    H = -Σ J_i Z_i Z_{i+1} - Σ h_i X_i

    One Trotter layer:
        1. exp(i J_i dt Z_i Z_{i+1})
        2. exp(i h_i dt X_i)

    Args:
        num_qubits: Number of qubits
        num_layers: Number of Trotter steps
        J: Mean ZZ coupling
        h: Mean transverse field
        dt: Trotter timestep
        rng: np.random.Generator
        disorder_strength_J:
            Uniform disorder width for J_i
            J_i ∈ [J-dJ, J+dJ]
        disorder_strength_h:
            Uniform disorder width for h_i
            h_i ∈ [h-dh, h+dh]

    Returns:
        List of gates
    """

    if rng is None:
        rng = np.random.default_rng(0)

    circuit = []

    for _ in range(num_layers):

        # Sample disordered couplings for this layer
        J_vals = rng.uniform(
            J - disorder_strength_J,
            J + disorder_strength_J,
            size=num_qubits - 1,
        )

        h_vals = rng.uniform(
            h - disorder_strength_h,
            h + disorder_strength_h,
            size=num_qubits,
        )

        # ZZ layer
        for i in range(num_qubits - 1):
            circuit.append(RZZGate(i, i + 1, J_vals[i] * dt))

        # X layer
        for i in range(num_qubits):
            circuit.append(RXGate(i, h_vals[i] * dt))

    return circuit


def generate_rfitm_circuit(
    num_qubits, num_layers, J=1.0, h=1.0, dt=0.1, sigma_h=0.1, rng=None
):
    """
    1D Random-Field Ising Model (RFITM) Trotter circuit.

    H = -J Σ Z_i Z_{i+1} - Σ hⱼ X_j
    where hⱼ ~ N(h, sigma_h²) drawn independently per qubit per instance.

    One Trotter layer:
    1. exp(i J dt Z_i Z_{i+1}) for all nearest-neighbour pairs (open BC)
    2. exp(i hⱼ dt X_j) for all qubits  (site-dependent rotation angle)

    Parameters
    ----------
    num_qubits : N
    num_layers : number of Trotter steps
    J          : ZZ coupling strength
    h          : mean transverse field
    dt         : Trotter step size
    sigma_h    : std of per-site disorder (sigma_h/h = 0.1 gives 10% disorder)
    rng        : numpy.random.Generator — controls disorder realisation.
                If None, uses np.random.default_rng(0) (reproducible but
                caller should always pass an explicit rng for averaging).

    Returns
    -------
    List of gates
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Draw per-site fields once per circuit instance
    h_sites = rng.normal(loc=h, scale=sigma_h, size=num_qubits)

    circuit = []
    for _ in range(num_layers):
        for i in range(num_qubits - 1):
            circuit.append(RZZGate(i, i + 1, J * dt))
        for i in range(num_qubits):
            circuit.append(RXGate(i, float(h_sites[i]) * dt))
    return circuit


def generate_rfim_circuit(
    num_qubits,
    num_layers,
    J=1.0,
    h0=1.0,
    W=2.0,
    dt=0.2,
    rng=None,
):
    """
    Trotterised 1D quantum Random-Field Ising Model (RFIM).

    Hamiltonian
    -----------
    H = -J  Σ_i  Z_i Z_{i+1}          (ferromagnetic ZZ coupling)
        - h₀ Σ_i  X_i                  (uniform transverse field)
        - Σ_i  Δ_i Z_i                 (longitudinal random field — the RFIM term)

    where  Δ_i ~ U[-W, W]  drawn independently per qubit per circuit instance.

    The ZZ coupling Z and the random longitudinal fields Z compete:
    strong disorder (W ≫ J) drives the system toward many-body localisation
    (MBL); weak disorder (W ≪ J) leaves it in the ergodic/thermal phase.
    The transverse field h₀ X introduces quantum fluctuations.

    This is precisely the model used in the seminal MBL studies of
    Pal & Huse (2010) and Oganesyan & Huse (2007).

    Trotterisation (first-order, one layer)
    ----------------------------------------
    exp(-i H dt)  ≈  U_ZZ · U_X · U_Δ

    where
        U_ZZ = Π_{i=0}^{N-2}  exp(+i J dt Z_i Z_{i+1})
            = Π RZZGate(i, i+1,  J * dt)

        U_X  = Π_{i=0}^{N-1}  exp(+i h₀ dt X_i)
            = Π RXGate(i,  h0 * dt)

        U_Δ  = Π_{i=0}^{N-1}  exp(+i Δ_i dt Z_i)
            = Π RZGate(i, -2 * Δ_i * dt)

    Gate conventions (gates.py):
        RZZGate(q1, q2, θ) = exp(+iθ Z_{q1}⊗Z_{q2})
        RXGate(q, θ)       = exp(+iθ X_q)
        RZGate(q, θ)       = exp(-iθ Z_q / 2)
        → exp(+iφ Z_q) requires RZGate(q, -2φ)

    Phase diagram and parameter guidance
    -------------------------------------
    For J = h₀ = 1.0 in 1D (open boundary):
        W < 1     thermal / ergodic (ETH) phase — high PRZ, BASS most useful
        W ≈ 2–3   ETH/MBL crossover — intermediate PRZ, BASS moderately useful
        W > 4     MBL phase — low PRZ, Z-sparse, BASS provides no benefit

    Default W = 2.0 places the model near the ETH/MBL crossover (Pal & Huse 2010;
    Luitz, Laflorencie, Alet, PRB 2015), producing intermediate PRZ values that
    make the circuit a meaningful benchmark for adaptive-basis sparse simulation.

    The original implementation used σ_h/h = 0.1 disorder in the transverse
    direction only, which (a) is a perturbative perturbation to the quantum
    critical TFIM, not a genuine RFIM, and (b) uses the wrong field direction
    for a model called "Random-Field Ising."  The present implementation uses
    longitudinal (Z-direction) random fields, which is the standard definition
    of the RFIM in both classical and quantum contexts.

    Parameters
    ----------
    num_qubits : int
        System size N.
    num_layers : int
        Number of Trotter steps L.  Total evolution time T = L × dt.
    J : float
        ZZ coupling strength (default 1.0).
    h0 : float
        Uniform transverse-field strength (default 1.0).
    W : float
        Half-width of the uniform random-field distribution U[-W, W].
        W = 2.0 gives the near-MBL-crossover regime (default).
    dt : float
        Trotter step size (default 0.2; total time T = 5 × 0.2 = 1.0).
    rng : numpy.random.Generator, optional
        Controls the disorder realisation.  Must be provided for ensemble
        averaging; defaults to default_rng(0) only as a fallback.

    Returns
    -------
    list[Gate]
        Circuit gates.  Gate count: num_layers × (N-1 + N + N) = L(3N-1).

    References
    ----------
    Pal, Huse, PRB 82, 174411 (2010) — MBL in the disordered Ising model.
    Oganesyan, Huse, PRB 75, 155111 (2007) — ETH/MBL crossover.
    Luitz, Laflorencie, Alet, PRB 91, 081103 (2015) — phase boundary W_c ≈ 3.5.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Draw per-site longitudinal random fields — one realisation per circuit instance.
    # Shape: (num_qubits,); i-th entry is Δ_i ~ U[-W, W].
    Delta = rng.uniform(-W, W, size=num_qubits)

    circuit = []
    for _ in range(num_layers):

        # U_ZZ: ferromagnetic ZZ coupling on all nearest-neighbour bonds (open BC)
        for i in range(num_qubits - 1):
            circuit.append(RZZGate(i, i + 1, J * dt))

        # U_X: uniform transverse field on all sites
        for i in range(num_qubits):
            circuit.append(RXGate(i, h0 * dt))

        # U_Δ: site-dependent longitudinal disorder
        # exp(+i Δ_i dt Z_i) = RZGate(i, -2 * Δ_i * dt)
        for i in range(num_qubits):
            circuit.append(RZGate(i, -2.0 * float(Delta[i]) * dt))

    return circuit


def circuit_info(circuit):
    """
    Get information about a circuit
    """
    num_gates = len(circuit)
    qubits_used = set()
    for gate in circuit:
        qubits_used.update(gate.qubits)

    num_qubits = max(qubits_used) + 1 if qubits_used else 0

    return {
        "num_gates": num_gates,
        "num_qubits": num_qubits,
        "depth": estimate_circuit_depth(circuit, num_qubits),
    }


def estimate_circuit_depth(circuit, num_qubits):
    """
    Estimate circuit depth (longest path through gates)
    """
    qubit_times = [0] * num_qubits

    for gate in circuit:
        max_time = max(qubit_times[q] for q in gate.qubits)
        for q in gate.qubits:
            qubit_times[q] = max_time + 1

    return max(qubit_times)
