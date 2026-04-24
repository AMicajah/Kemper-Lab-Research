import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Parameters
N = 2
Jx = 1.0
Jy = 0.5
Jz = 0.8

tlist = np.linspace(0, 10, 100)
dt = tlist[1] - tlist[0]

shots = 5000

# Dissipation rates
gamma1 = 1.0
gamma2 = 0.25

# Heisenberg Trotter Step
def heisenberg_step(qc, Jx, Jy, Jz, dt):
    # ZZ
    qc.cx(0,1)
    qc.rz(2 * Jz * dt, 1)
    qc.cx(0,1)

    # XX
    qc.h(0)
    qc.h(1)
    qc.cx(0,1)
    qc.rz(2 * Jx * dt, 1)
    qc.cx(0,1)
    qc.h(0)
    qc.h(1)

    # YY
    qc.sdg(0)
    qc.sdg(1)
    qc.h(0)
    qc.h(1)
    qc.cx(0,1)
    qc.rz(2 * Jy * dt, 1)
    qc.cx(0,1)
    qc.h(0)
    qc.h(1)
    qc.s(0)
    qc.s(1)


sim = AerSimulator()

# Observable extraction
def compute_sz(counts, qubit_index, shots):
    exp = 0
    for bitstring, count in counts.items():
        bit = int(bitstring[qubit_index])
        exp += (1 if bit == 0 else -1) * count
    return exp / shots

# Time evolution
sz_exp1 = []
sz_exp2 = []
def sigma_minus_channel(qc, system, ancilla, gamma, dt):
    p = gamma * dt
    theta = 2 * np.arcsin(np.sqrt(p))
    
    qc.reset(ancilla)
    
    # sigma minus circuit
    qc.x(system)
    qc.cry(theta, system, ancilla)
    qc.x(system)
    
    qc.cx(ancilla, system)
    
    qc.reset(ancilla)
    
for step in range(len(tlist)):
    
    qc = QuantumCircuit(N + 2, N)
    sys0, sys1 = 0, 1
    anc0, anc1 = 2, 3
    
    # Initial state |10>
    qc.x(sys0)
    
    # Apply time evolution
    for _ in range(step):
        heisenberg_step(qc, Jx, Jy, Jz, dt)
    
        # trigger noise
        sigma_minus_channel(qc, sys0, anc0, gamma1, dt)
        sigma_minus_channel(qc, sys1, anc1, gamma2, dt)
        
    qc.measure([sys0, sys1], [1, 0])
    
    
    tqc = transpile(qc, sim, optimization_level=0)
    
    result = sim.run(
        tqc,
        shots=shots
    ).result()
    
    counts = result.get_counts()
    #print(counts)
    sz_exp1.append(compute_sz(counts, 0, shots))
    sz_exp2.append(compute_sz(counts, 1, shots))

# Plot
plt.figure(figsize=(7,5))
plt.plot(tlist, sz_exp1, 'b-', label='Site 1 <σ_z>')
plt.plot(tlist, sz_exp2, 'r-', label='Site 2 <σ_z>')
plt.xlabel('Time')
plt.ylabel('<σ_z>')
plt.title('Two-Site Dissipative Heisenberg (Shot-Based)')
plt.legend()
plt.tight_layout()
plt.show()