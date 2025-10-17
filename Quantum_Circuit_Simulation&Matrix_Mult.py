from qiskit import QuantumCircuit
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np

# Qiskit Sim -----------------------------------------------------------------------------------------
Gamma = 1
Delta_t = 0.1
beta = 5
k = np.pi/4
Omega = 2     
timesteps = 100

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

simulator = Aer.get_backend("qasm_simulator")
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
plt.plot(range(timesteps), probs_sim, "o-", markersize=4, linewidth=1, label="Qiskit sim")
plt.plot(range(timesteps), probs_matrix, "--", linewidth=1.5, label="Matrix mult")
plt.xlabel("Time Step")
plt.ylabel("Probability P(|1⟩)")
plt.title("Time evolution of electron density")
plt.ylim(0, 1)
plt.legend()
plt.show()
