import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, Operator
from qiskit_aer.primitives import Estimator as AerEstimator
import time
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Coupling Amplitudes
J_XY = 1.0
JZ_values = np.linspace(-2, 2, 20)
L = 8 # system size
tlist = np.linspace(0, 200, 200)
trotter_slices = 4
shots = 500

# Shot-based sim ---------------------------------------------------------------------------------------

dt_full = tlist[1] - tlist[0]
dt_small = dt_full / trotter_slices

estimator = AerEstimator()

def build_hamiltonian_op(L, J_XY, J_Z):
    paulis = []
    coeffs = []

    for i in range(L-1):
        # XX
        label = ['I']*L
        label[L-1-i], label[L-2-i] = 'X','X'
        paulis.append(''.join(label))
        coeffs.append(J_XY)

        # YY
        label = ['I']*L
        label[L-1-i], label[L-2-i] = 'Y','Y'
        paulis.append(''.join(label))
        coeffs.append(J_XY)

        # ZZ
        label = ['I']*L
        label[L-1-i], label[L-2-i] = 'Z','Z'
        paulis.append(''.join(label))
        coeffs.append(J_Z)

    return SparsePauliOp(paulis, coeffs)

    
def build_trotter_step(J_XY, J_Z):
    qc = QuantumCircuit(L)

    for i in range(L-1):
        
        # XX
        theta = J_XY * dt_small
        qc.h(i); qc.h(i+1)
        qc.cx(i, i+1)
        qc.rz(2*theta, i+1)
        qc.cx(i, i+1)
        qc.h(i); qc.h(i+1)

        # YY
        qc.rx(np.pi/2, i); qc.rx(np.pi/2, i+1)
        qc.cx(i, i+1)
        qc.rz(2*theta, i+1)
        qc.cx(i, i+1)
        qc.rx(-np.pi/2, i); qc.rx(-np.pi/2, i+1)

        # ZZ
        theta_Z = J_Z * dt_small
        qc.cx(i, i+1)
        qc.rz(2*theta_Z, i+1)
        qc.cx(i, i+1)

    return qc
    
def make_Z_op(L, i):
    label = ['I'] * L
    label[L-1-i] = 'Z'
    return SparsePauliOp(''.join(label), [1.0])

def make_ZZ_op(L, i, j):
    label = ['I'] * L
    label[L-1-i] = 'Z'
    label[L-1-j] = 'Z'
    return SparsePauliOp(''.join(label), [1.0])

def run_qiskit_trotter_estimator(J_XY, J_Z, L, tlist, trotter_slices):

    n_steps = len(tlist) * trotter_slices

    hamiltonian_op = build_hamiltonian_op(L, J_XY, J_Z)
    trotter_step = build_trotter_step(J_XY, J_Z)
    
    initial = QuantumCircuit(L)
    # Neel state
    #|01010101>
    for i in range(L):
        if i % 2 == 1:
            initial.x(i)

    evolved = initial.compose(trotter_step.repeat(n_steps))

    observables = [hamiltonian_op]

     # Single Z for each qubit
    for i in range(L):
        observables.append(make_Z_op(L, i))
    # ZZ for nearest-neighbor pair
    for i in range(L-1):
        observables.append(make_ZZ_op(L, i, i+1))

    circuits = [evolved.copy() for _ in observables]
    
    # Run estimator
    job = estimator.run(
        circuits=circuits,
        observables=observables,
        shots=shots
    )

    vals = job.result().values

    energy = vals[0]

    # Connected Correlation
    idx_Z_start = 1
    idx_ZZ_start = 1 + L

    corr_sum = 0.0
    for i in range(L-1):
        ZiZj = vals[idx_ZZ_start + i]
        Zi = vals[idx_Z_start + i]
        Zj = vals[idx_Z_start + i + 1]
        corr_sum += ZiZj - Zi * Zj

    corr = corr_sum / (L - 1)

    return energy, corr

energy_vs_JZ = []
corr_vs_JZ = []
start_total = time.time()

for J_Z_idx, J_Z in enumerate(JZ_values):
    start_jz = time.time()

    energy, corr = run_qiskit_trotter_estimator(J_XY, J_Z, L, tlist, trotter_slices)

    energy_vs_JZ.append(energy)
    corr_vs_JZ.append(corr)

    print(f"J_Z={J_Z:.2f} done in {time.time() - start_jz:.2f}s")

print(f"Total runtime for all J_Z: {time.time() - start_total:.2f}s")

# Analytical statevector sim ------------------------------------------------------------------------------------------

def run_qiskit_exact_statevector(J_XY, J_Z, L, tlist):

    t_final = tlist[-1]

    # Build Hamiltonian
    hamiltonian_op = build_hamiltonian_op(L, J_XY, J_Z)

    H_mat = hamiltonian_op.to_matrix()

    # Build Unitary
    U = Operator(expm(-1j * H_mat * t_final))

    # Neel state
    initial = QuantumCircuit(L)
    for i in range(L):
        if i % 2 == 1:
            initial.x(i)

    psi0 = Statevector.from_instruction(initial)

    psi_t = psi0.evolve(U)

    energy = np.real(psi_t.expectation_value(hamiltonian_op))

    # Correlations
    Z_ops = [make_Z_op(L, i) for i in range(L)]
    ZZ_ops = [make_ZZ_op(L, i, i+1) for i in range(L-1)]

    Zi_vals = [np.real(psi_t.expectation_value(Z)) for Z in Z_ops]
    ZiZj_vals = [np.real(psi_t.expectation_value(ZZ)) for ZZ in ZZ_ops]

    corr_sum = 0.0
    for i in range(L-1):
        corr_sum += ZiZj_vals[i] - Zi_vals[i] * Zi_vals[i+1]

    corr = corr_sum / (L - 1)

    return energy, corr

energy_exact = []
corr_exact = []

for J_Z in JZ_values:
    energy, corr = run_qiskit_exact_statevector(J_XY, J_Z, L, tlist)
    energy_exact.append(energy)
    corr_exact.append(corr)

# Plots-------------------------------------------------------------------------------------------------

plt.figure()
plt.plot(JZ_values, energy_vs_JZ, 'o-', label="Shot-based")
plt.plot(JZ_values, energy_exact, 's--', label="Exact statevector")
plt.xlabel("J_Z / J_XY")
plt.ylabel("Total Energy ⟨H⟩ at final time")
plt.title("Phase-like plot of XXZ chain")
plt.legend()
plt.show()

plt.figure()
plt.plot(JZ_values, corr_vs_JZ, 'o-', label="Shot-based (Trotter")
plt.plot(JZ_values, corr_exact, 's--', label="Exact (statevector)")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(1.0, color='r', linestyle='--', linewidth=0.8)
plt.axvline(-1.0, color='r', linestyle='--', linewidth=0.8)
plt.xlabel("J_Z / J_XY")
plt.ylabel(r"$\langle Z_i Z_{i+1} \rangle - \langle Z_i \rangle\langle Z_{i+1} \rangle$")
plt.title("Spin-spin correlation vs anisotropy")
plt.legend()
plt.show()
