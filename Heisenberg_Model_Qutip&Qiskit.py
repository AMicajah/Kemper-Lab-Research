import numpy as np
from qutip import tensor, basis, sigmax, sigmay, sigmaz, qeye, sesolve, expect
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector, Pauli
import matplotlib.pyplot as plt

J = -1.0  # coupling amplitude
L = 2     # system size
tlist = np.linspace(0, 30, 101)

# QuTiP simulation -------------------------------------------------------------------------------
def run_qutip(J, L, tlist):
    
    # Define spin up and down states
    up = basis(2, 0)
    down = basis(2, 1)
    
    # Defining Pauli operators
    Id_qutip = qeye(2)
    X_qutip = sigmax()
    Y_qutip = sigmay()
    Z_qutip = sigmaz()
    
    # Initialize Hamiltonian
    H_qutip = 0
    
    for i in range(L-1):
        for op1, op2 in zip([X_qutip, Y_qutip, Z_qutip], [X_qutip, Y_qutip, Z_qutip]):
            op_list = [Id_qutip]*L
            op_list[i] = op1
            op_list[i+1] = op2
            H_qutip += J * tensor(op_list)
    
    # Diagonalize Hamiltonian to find its ground state
    w, v = H_qutip.eigenstates(eigvals=1)
    
    # Initial state: all up
    psi0 = tensor([up]*L)
    
    # Perform time evolution
    output = sesolve(H_qutip, psi0, tlist)
    
    # Retrieve time-evolved states
    states = output.states
    
    # Measure energy
    energies_qutip = [expect(H_qutip, state) for state in states]
    
    return energies_qutip

# Qiskit Trotter simulation ---------------------------------------------------------------------------

def run_qiskit(J, L, tlist):
    steps = len(tlist)
    dt = tlist[1] - tlist[0]
    
    # Build terms for energy measurement
    terms = []
    for i in range(L-1):
        for pauli in ['XX','YY','ZZ']:
            label = ['I']*L
            label[i], label[i+1] = pauli[0], pauli[1]
            terms.append((J, Pauli(''.join(label))))
    
    def measure_energy(psi):
        return sum(c * psi.expectation_value(p).real for c,p in terms)
    
    # Single Trotter step circuit
    def trotter_step_circuit(L, J, dt):
        qc = QuantumCircuit(L)
        for i in range(L-1):
            # XX interaction
            qc.cx(i, i+1)
            qc.rx(2*J*dt, i+1) 
            qc.cx(i, i+1)
            
            # YY interaction 
            qc.cx(i, i+1)
            qc.ry(2*J*dt, i+1)
            qc.cx(i, i+1)
            
            # ZZ interaction
            qc.cx(i, i+1)
            qc.rz(2*J*dt, i+1)
            qc.cx(i, i+1)
            
        return qc
    
    # Initialize statevector
    psi_qiskit = Statevector.from_label('0'*L)
    energies_qiskit = []
    
    # Run Trotter evolution
    for _ in range(steps):
        qc = trotter_step_circuit(L, J, dt)
        if _ < 2:
            print(qc) 
        psi_qiskit = psi_qiskit.evolve(qc)
        energies_qiskit.append(measure_energy(psi_qiskit))
    
    return energies_qiskit

# Return Energies ---------------------------------------------------------------------
energies_qutip = run_qutip(J, L, tlist)
energies_qiskit = run_qiskit(J, L, tlist)


# Plot --------------------------------------------------------------------------------
plt.plot(tlist, energies_qutip, label='QuTiP (Exact)', linewidth=2)
plt.plot(tlist, energies_qiskit, '--', label='Qiskit (Trotterized)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Total Energy <H>')
plt.title('Energy: QuTiP vs Qiskit')
plt.legend()
plt.show()
