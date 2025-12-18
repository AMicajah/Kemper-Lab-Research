import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp
import time
import matplotlib.pyplot as plt

# Coupling Amplitudes
J_XY = 1.0
JZ_values = np.linspace(-2, 2, 20)
L = 8 # system size
tlist = np.linspace(0, 20, 25)
trotter_slices = 4
shots = 200

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

def run_qiskit_trotter_estimator(J_XY, J_Z, L, tlist, trotter_slices):
    hamiltonian_op = build_hamiltonian_op(L, J_XY, J_Z)
    trotter_step = build_trotter_step(J_XY, J_Z)

    energies = []
    evolved = QuantumCircuit(L)

    for t_idx, t in enumerate(tlist):
        t0 = time.time()
        
        # Trotter evolution
        for _ in range(trotter_slices):
            evolved.compose(trotter_step, inplace=True)
        t_trotter = time.time() - t0

        t1 = time.time()
        
        # Estimator call
        job = estimator.run(circuits=evolved, observables=hamiltonian_op, shots=shots)
        energy = job.result().values[0]
        t_estimator = time.time() - t1

        energies.append(energy)

        #print(f"Time step {t_idx}: Trotter {t_trotter:.3f}s, Estimator {t_estimator:.3f}s")

    return energies


energy_vs_JZ = []

start_total = time.time()
for J_Z_idx, J_Z in enumerate(JZ_values):
    start_jz = time.time()
    energies = run_qiskit_trotter_estimator(J_XY, J_Z, L, tlist, trotter_slices)
    energy_vs_JZ.append(energies[-1])
    print(f"J_Z={J_Z:.2f} done in {time.time() - start_jz:.2f}s")
print(f"Total runtime for all J_Z: {time.time() - start_total:.2f}s")
    
plt.plot(JZ_values, energy_vs_JZ, 'o-')
plt.xlabel("J_Z / J_XY")
plt.ylabel("Total Energy <H> at final time")
plt.title("Phase-like plot of XXZ chain")
plt.show()
