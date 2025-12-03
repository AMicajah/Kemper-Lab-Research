import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
import matplotlib.pyplot as plt

# Coupling Amplitudes
J_XY = 1.0
JZ_values = np.linspace(-2, 2, 20)

L = 3 # system size
tlist = np.linspace(0, 20, 50)
trotter_slices = 2

def run_qiskit_trotter_shotbased(J_XY, J_Z, L, tlist, trotter_slices=5, shots=500):

    dt_full = tlist[1] - tlist[0]
    dt_small = dt_full / trotter_slices

    sim = AerSimulator()

    def build_hamiltonian_terms(L, J_XY, J_Z):
        terms = []
        for i in range(L-1):
            # XX and YY terms
            for p in ['XX','YY']:
                label = ['I']*L
                label[L-1-i], label[L-2-i] = p[0], p[1]
                terms.append((J_XY, Pauli(''.join(label))))
            # ZZ term
            label = ['I']*L
            label[L-1-i], label[L-2-i] = 'Z','Z'
            terms.append((J_Z, Pauli(''.join(label))))
        return terms
    
    # Build one Trotter slice
    def build_trotter_step():
        qc = QuantumCircuit(L)
        theta = J_XY * dt_small
        for i in range(L-1):

            # XX
            qc.h(i); qc.h(i+1)
            qc.cx(i,i+1)
            qc.rz(2*theta, i+1)
            qc.cx(i,i+1)
            qc.h(i); qc.h(i+1)

            # YY
            qc.rx(np.pi/2, i); qc.rx(np.pi/2, i+1)
            qc.cx(i,i+1)
            qc.rz(2*theta, i+1)
            qc.cx(i,i+1)
            qc.rx(-np.pi/2, i); qc.rx(-np.pi/2, i+1)

            # ZZ
            theta_Z = J_Z * dt_small
            qc.cx(i,i+1)
            qc.rz(2*theta_Z, i+1)
            qc.cx(i,i+1)

        return qc

    # Hamiltonian terms for this set of couplings
    terms = build_hamiltonian_terms(L, J_XY, J_Z)

    # Trotter slice
    trotter_step = build_trotter_step()

    # Expectation estimator
    def estimate_pauli(pauli, evolved):
        qc = evolved.copy()

        for q, axis in enumerate(str(pauli)[::-1]):
            if axis == 'X':
                qc.h(q)
            elif axis == 'Y':
                qc.sdg(q); qc.h(q)

        qc.measure_all()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()

        exp = 0
        for bitstring, count in counts.items():
            parity = 1
            for q, axis in enumerate(str(pauli)[::-1]):
                if axis != 'I' and bitstring[::-1][q] == '1':
                    parity *= -1
            exp += parity * (count/shots)

        return exp

    # time evolution
    energies = []
    evolved = QuantumCircuit(L)   # Start in |000…0>
    
    for step_idx, t in enumerate(tlist):

        # Add one delta t of evolution
        for _ in range(trotter_slices):
            evolved = evolved.compose(trotter_step)

        # Measure <H>
        E = 0
        for coeff, p in terms:
            E += coeff * estimate_pauli(p, evolved)

        energies.append(E)

    return energies

energy_vs_JZ = []
for J_Z in JZ_values:
    energies = run_qiskit_trotter_shotbased(J_XY, J_Z, L, tlist, trotter_slices=2, shots=500)
    energy_vs_JZ.append(energies[-1])
    
plt.plot(JZ_values, energy_vs_JZ, 'o-')
plt.xlabel("J_Z / J_XY")
plt.ylabel("Total Energy <H> at final time")
plt.title("Phase-like plot of XXZ chain")
plt.show()
