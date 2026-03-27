from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
import matplotlib.pyplot as plt

n = 3
theta_vals = np.linspace(0, np.pi, 20)
sim = AerSimulator()
shots = 10000
    
hadamard_vals = []
exact_vals = []

for theta in theta_vals:
    A = QuantumCircuit(n)
    B = QuantumCircuit(n)
    
    # A: some rotations + entanglement
   # for i in range(n):
       # A.ry(np.pi/4, i)
  #  for i in range(n-1):
  #      A.cx(i, i+1)
    
    # B: different state
   # for i in range(n):
    #    B.rx(np.pi/3, i)
   # for i in range(n-1):
    #    B.cx(i, i+1)
    
    # --- A ---
    for i in range(n):
        A.ry(np.pi/4, i)
        A.rz(np.pi/5, i)
    
    for i in range(n-1):
        A.cx(i, i+1)
    
    for i in range(n):
        A.ry(np.pi/6, i)
    
    # --- B ---
    for i in range(n):
        B.rx(theta, i)
        B.rz(theta/2, i)
    
    for i in range(n-1):
        B.cx(i, i+1)
    
    for i in range(n):
        B.ry(theta/3, i)
    
    # B^{dagger}A
    BA = B.inverse().compose(A)
    
    # Hadamard test circuit
    qc = QuantumCircuit(n+1,1)
    
    anc = 0
    system = list(range(1, n+1))
    
    U = BA.to_gate()
    CU = U.control()
    
    qc.h(anc)
    qc.append(CU, [anc] + system)
    qc.h(anc)
    
    qc.measure(anc, 0)
    
    compiled = transpile(qc, sim)

    result = sim.run(compiled, shots=shots).result()
    
    counts = result.get_counts()
    
    # Probability ancilla = 0
    p0 = counts.get('0',0) / shots
    
    overlap_est = 2*p0 - 1
    hadamard_vals.append(overlap_est)
    
    # Exact overlap
    psi_state = Statevector.from_label('0'*n).evolve(A)
    phi_state = Statevector.from_label('0'*n).evolve(B)
    
    overlap_exact = phi_state.inner(psi_state)
    exact_vals.append(np.real(overlap_exact)) 
    
    #print("theta:", theta)
   # print("Exact:", np.real(overlap_exact))
    #print("Hadamard:", overlap_est)
   # print()
    
plt.figure(figsize=(8,5))

plt.plot(theta_vals, exact_vals, label='Exact', linewidth=2)
plt.scatter(theta_vals, hadamard_vals, label='Hadamard (shots)', s=30, color='red')

plt.xlabel("theta")
plt.ylabel("Re(<phi|psi>)")
plt.title(f"Hadamard Test vs Exact (n={n})")
plt.legend()

plt.ylim(min(min(hadamard_vals), min(exact_vals)) - 1,
         max(max(hadamard_vals), max(exact_vals)) + 1)

plt.show()