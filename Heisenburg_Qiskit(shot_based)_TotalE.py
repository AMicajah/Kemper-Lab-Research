import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
import matplotlib.pyplot as plt

J = -1.0  # coupling amplitude
L = 3 # system size
tlist = np.linspace(0, 20, 100)
trotter_slices = 5

def run_qiskit_trotter_shotbased(J, L, tlist, trotter_slices=5, shots=2000):

    dt_full = tlist[1] - tlist[0]
    dt_small = dt_full / trotter_slices

    sim = AerSimulator()

    # Hamiltonian terms
    terms = []
    for i in range(L-1):
        for p in ['XX','YY','ZZ']:
            label = ['I'] * L
            label[L-1-i], label[L-2-i] = p[0], p[1]
            terms.append((J, Pauli(''.join(label))))

    # Build one Trotter slice
    def build_trotter_step():
        qc = QuantumCircuit(L)
        theta = J * dt_small
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
            qc.cx(i,i+1)
            qc.rz(2*theta, i+1)
            qc.cx(i,i+1)

        return qc

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

        # Add one Δt of evolution
        for _ in range(trotter_slices):
            evolved = evolved.compose(trotter_step)

        # Measure <H>
        E = 0
        for coeff, p in terms:
            E += coeff * estimate_pauli(p, evolved)

        energies.append(E)

    return energies

energies = run_qiskit_trotter_shotbased(J, L, tlist, trotter_slices=5, shots=2000)
plt.plot(tlist,energies)
