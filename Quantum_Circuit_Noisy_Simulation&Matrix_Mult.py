import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, pauli_error
from qiskit_aer import AerSimulator

# Noise Model --------------------------------------------------------------------------------------------

noise_model = NoiseModel()

# --- Pauli error: 2% X, 1% Z ---
pauli_1q = pauli_error([
    ("X", 0.005),
    ("Z", 0.005),
    ("I", 0.99)
])

# 2-qubit version for CX/CRX gates
pauli_2q = pauli_error([
    ("IX", 0.01),
    ("XI", 0.01),
    ("XX", 0.005),
    ("II", 0.975)
])

# single-qubit depolarizing for rx, ry, rz
depol_1q = depolarizing_error(0.05, 1)

# two-qubit depolarizing for cx and crx
depol_2q = depolarizing_error(0.05, 2)

combined_1q = depol_1q.compose(pauli_1q)
combined_2q = depol_2q.compose(pauli_2q)

noise_model.add_all_qubit_quantum_error(combined_1q, ['rx', 'ry', 'rz'])
noise_model.add_all_qubit_quantum_error(combined_2q, ['cx', 'crx'])

# readout error
readout_err = ReadoutError([[0.98, 0.02], 
                            [0.02, 0.98]])
noise_model.add_all_qubit_readout_error(readout_err)

simulator = AerSimulator(noise_model=noise_model)


# Qiskit Sim -----------------------------------------------------------------------------------------

Gamma = 1
Delta_t = 0.1
beta = 5
k = np.pi/4
Omega = 2     
timesteps = 10

theta_list = []
phi_list = []

for i in range(timesteps):
    e_i = -2 * np.cos(k + Omega * i * Delta_t)
    
    value_theta = (Gamma * Delta_t) / (np.exp(beta * e_i) + 1)
    value_phi = (Gamma * Delta_t) / (np.exp(-beta * e_i) + 1)
    
    theta_i = 2 * np.arcsin(np.sqrt(value_theta))
    phi_i = 2 * np.arcsin(np.sqrt(value_phi))
    
    theta_list.append(theta_i)
    phi_list.append(phi_i)

shots = 1000

probs_sim = []

for j in range(timesteps):
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    for i in range(j+1):
        qc.rx(theta_list[i], 0)
        qc.crx(phi_list[i] - theta_list[i], 1, 0)
        qc.cx(0, 1)
        qc.reset(0)
    qc.measure(1, 0)
    
    job = simulator.run(qc, shots=1000)
    counts = job.result().get_counts()
    p1 = counts.get("1", 0) / 1000
    probs_sim.append(p1)

# Analytical Matrix Multiplication----------------------------------------------------------------------------------------------

def Rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Rx_on_ancilla(theta):
    return np.kron(np.eye(2), Rx(theta))   

def CRx_control_system(phi):
    P0_sys = np.array([[1,0],[0,0]], dtype=complex)
    P1_sys = np.array([[0,0],[0,1]], dtype=complex)
    I_anc = np.eye(2, dtype=complex)
    Rx_anc = Rx(phi)
    return np.kron(P0_sys, I_anc) + np.kron(P1_sys, Rx_anc)

def CNOT_control_ancilla():
    P0_anc = np.array([[1,0],[0,0]], dtype=complex)
    P1_anc = np.array([[0,0],[0,1]], dtype=complex)
    I_sys = np.eye(2, dtype=complex)
    X_sys = np.array([[0,1],[1,0]], dtype=complex)
    return np.kron(I_sys, P0_anc) + np.kron(X_sys, P1_anc)

def evolve_one_step(rho_sys, theta, phi_minus_theta):
    # ancilla in |0><0|
    anc0 = np.array([1,0], dtype=complex)
    rho_anc = np.outer(anc0, anc0.conj())
    rho_joint = np.kron(rho_sys, rho_anc)

    U = CNOT_control_ancilla() @ CRx_control_system(phi_minus_theta) @ Rx_on_ancilla(theta)
    rho_joint = U @ rho_joint @ U.conj().T

    rho_joint = rho_joint.reshape(2,2,2,2)
    rho_sys_next = np.trace(rho_joint, axis1=1, axis2=3)
    return rho_sys_next

probs_matrix = []

for j in range(timesteps):
    rho_sys = np.array([[0,0],[0,1]], dtype=complex)  
    for i in range(j+1):
        theta = theta_list[i]
        phi_minus_theta = phi_list[i] - theta_list[i]
        rho_sys = evolve_one_step(rho_sys, theta, phi_minus_theta)
    probs_matrix.append(np.real_if_close(rho_sys[1,1]))

plt.figure(figsize=(14, 6))
plt.plot(range(timesteps), probs_matrix, "--", linewidth=1.5, label="Matrix mult")
plt.plot(range(timesteps), probs_sim, "o-", markersize=4, linewidth=1, label="Qiskit sim (Noisy)")

plt.fill_between(range(timesteps),
                 np.minimum(probs_sim, probs_matrix),
                 np.maximum(probs_sim, probs_matrix),
                 color='gray', alpha=0.3, label='Deviation region')

plt.xlabel("Time Step")
plt.ylabel("Probability P(|1⟩)")
plt.title("Time evolution of electron density")
plt.ylim(0, 1)
plt.legend()
plt.show()