from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np

theta = np.pi/3

# A prepares |psi> = |0>
A = QuantumCircuit(1)

# B prepares |phi> = RY(theta)|0>
B = QuantumCircuit(1)
B.ry(theta, 0)

# B^{dagger}A
BA = B.inverse().compose(A)

# Hadamard test circuit
qc = QuantumCircuit(2,1)

anc = 0
sys = 1

qc.h(anc)

qc.append(BA.control(), [anc, sys])

qc.h(anc)

qc.measure(anc, 0)

# Run shot-based sim
sim = AerSimulator()

compiled = transpile(qc, sim)

shots = 10000
result = sim.run(compiled, shots=shots).result()

counts = result.get_counts()

# Probability ancilla = 0
p0 = counts.get('0',0) / shots

overlap_est = 2*p0 - 1

print("Hadamard test estimate:", overlap_est)

# Exact overlap
psi_state = Statevector.from_label('0').evolve(A)
phi_state = Statevector.from_label('0').evolve(B)

overlap_exact = phi_state.inner(psi_state)

print("Exact overlap:", overlap_exact)
