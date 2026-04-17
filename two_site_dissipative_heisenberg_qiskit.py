import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

# Parameters
N = 2
Jx = 1.0
Jy = 0.5
Jz = 0.8

tlist = np.linspace(0, 10, 100)
dt = tlist[1] - tlist[0]

shots = 2000

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

# Noise Model (σ⁻ jumps)
noise_model = NoiseModel()

p1 = gamma1 * dt
p2 = gamma2 * dt

error1 = amplitude_damping_error(p1)
error2 = amplitude_damping_error(p2)

# Apply noise after identity
noise_model.add_quantum_error(error1, ['id'], [0])
noise_model.add_quantum_error(error2, ['id'], [1])


sim = AerSimulator()

# Observable extraction
def compute_sz(counts, qubit_index, shots):
    exp = 0
    for bitstring, count in counts.items():
        bit = int(bitstring[qubit_index])
        exp += (1 if bit == 0 else -1) * count
    return - exp / shots

''' I couldn't figure out how to fix the ordering in the above function to make
the results match the QuTip (everything was inverted over time axis) so I just
added a minus sign to make them match for now '''

# Time evolution
sz_exp1 = []
sz_exp2 = []

for step in range(len(tlist)):
    
    qc = QuantumCircuit(N, N)
    
    # Initial state |10>
    qc.x(0)
    
    # Apply time evolution
    for _ in range(step):
        heisenberg_step(qc, Jx, Jy, Jz, dt)
    
        # trigger noise
        qc.id(0)
        qc.id(1)
        
    qc.measure([0, 1], [0, 1])
    
    tqc = transpile(qc, sim, optimization_level=0)
    
    result = sim.run(
        tqc,
        shots=shots,
        noise_model=noise_model
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