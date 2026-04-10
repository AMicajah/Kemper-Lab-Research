from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

n = 3 
theta_vals = np.linspace(0, np.pi, 20)  
sim = AerSimulator()
shots = 5000

swap_vals = []
exact_vals = []

for theta in theta_vals:
    A = QuantumCircuit(n)
    B = QuantumCircuit(n)
    
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
    
    # SWAP test circuit
    qc = QuantumCircuit(1 + 2*n, 1)  # ancilla + two n-qubit registers
    
    anc = 0
    reg_psi = list(range(1, n+1))
    reg_phi = list(range(n+1, 2*n+1))
    
    qc.h(anc)
    
    # Controlled SWAPs
    for i in range(n):
        qc.cswap(anc, reg_psi[i], reg_phi[i])
    
    qc.h(anc)
    qc.measure(anc, 0)
    
    sv = Statevector.from_label('0'*n)
    psi_state = sv.evolve(A)
    phi_state = sv.evolve(B)
    
    qc_init = QuantumCircuit(1 + 2*n)
    qc_init.append(A.to_gate(), reg_psi)
    qc_init.append(B.to_gate(), reg_phi)
    
    # Add SWAP test to initial circuit
    qc_total = qc_init.compose(qc, front=False)
    # print(qc_total)
    # Run sim
    
    tqc = transpile(qc_total, sim, optimization_level=3)

    print("Depth:", tqc.depth())
    print("CNOTs:", tqc.count_ops().get("cx", 0))
    
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    
    p0 = counts.get('0', 0) / shots
    overlap_squared = 2*p0 - 1
    swap_vals.append(overlap_squared)
  #  print("SWAP:", overlap_squared)
    
    overlap_exact = np.abs(phi_state.inner(psi_state))**2
    exact_vals.append(overlap_exact)
   # print("Exact:", overlap_exact)

plt.figure(figsize=(8,5))

plt.plot(theta_vals, exact_vals, label='Exact', linewidth=2)
plt.scatter(theta_vals, swap_vals, label='SWAP (shots)', s=30, color='red')

plt.xlabel("theta")
plt.ylabel("|<phi|psi>|^2")
plt.title(f"SWAP Test vs Exact (n={n})")
plt.legend()

plt.ylim(min(min(swap_vals), min(exact_vals)) - 1,
         max(max(swap_vals), max(exact_vals)) + 1)

plt.show()