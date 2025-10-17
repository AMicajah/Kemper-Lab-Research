from qiskit import QuantumCircuit
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np


Gamma = 2
Delta_t = 0.1
beta = 5
k = np.pi/4
Omega = 5   
timesteps = 50

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

#print(theta_list)
#print(phi_list)

simulator = Aer.get_backend("qasm_simulator")
probs = []

for j in range(timesteps):
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    for i in range(j+1):
        qc.rx(theta_list[i], 0)
        qc.crx(phi_list[i] - theta_list[i], 1, 0)
        qc.cx(0, 1)
        qc.reset(0)
    qc.measure(1, 0)

    #print(qc)
    
    job = simulator.run(qc, shots=1000)
    counts = job.result().get_counts()
    p1 = counts.get("1", 0) / 1000
    probs.append(p1)

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(range(timesteps), probs, "o-", markersize=4, linewidth=1)
plt.xlabel("Time Step")
plt.ylabel("Probability P(|1⟩)")
plt.title("Time evolution of electron density")
plt.ylim(0, 1)