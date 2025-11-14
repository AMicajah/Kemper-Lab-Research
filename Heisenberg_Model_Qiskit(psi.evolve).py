import numpy as np
from qutip import tensor, basis, sigmax, sigmay, sigmaz, qeye, sesolve, expect
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector, Pauli
import matplotlib.pyplot as plt

J = 1.0  # coupling amplitude
L = 5
# system size
tlist = np.linspace(0, 30, 100)
trotter_slices = 5


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

def run_qiskit(J, L, tlist, trotter_slices):
    dt_full = tlist[1] - tlist[0]
    dt_small = dt_full / trotter_slices

    # Build energy measurement terms
    terms = []
    for i in range(L-1):
        for p in ['XX','YY','ZZ']:
            label = ['I']*L
            label[L-1-i], label[L-2-i] = p[0], p[1]
            terms.append((J, Pauli(''.join(label))))

    def measure_energy(psi):
        return sum(coeff * np.real(psi.expectation_value(Operator(p))) for coeff, p in terms)

    psi = Statevector.from_label('0'*L)
    energies = []

    #psi_test = Statevector.from_label('0'*L)
    #E_test = sum(coeff*np.real(psi_test.expectation_value(Operator(p))) for coeff, p in terms)
    #print("Sanity check: <H> for |00> =", E_test)

    for t in tlist:
        for _ in range(trotter_slices):
            qc = QuantumCircuit(L)
            for i in range(L-1):
                theta = J * dt_small
                # XX rotation
                qc.h(i); qc.h(i+1)
                qc.cx(i,i+1)
                qc.rz(2*theta, i+1)
                qc.cx(i,i+1)
                qc.h(i); qc.h(i+1)
                # YY rotation
                qc.rx(np.pi/2, i); qc.rx(np.pi/2, i+1)
                qc.cx(i,i+1)
                qc.rz(2*theta, i+1)
                qc.cx(i,i+1)
                qc.rx(-np.pi/2, i); qc.rx(-np.pi/2, i+1)
                # ZZ rotation
                qc.cx(i,i+1)
                qc.rz(2*theta, i+1)
                qc.cx(i,i+1)
            psi = psi.evolve(qc)
        energies.append(measure_energy(psi))
    #print(qc)
    return energies

# Return Energies ---------------------------------------------------------------------
energies_qutip = run_qutip(J, L, tlist)
energies_qiskit = run_qiskit(J, L, tlist, trotter_slices)
energies_qiskit = [round(e, 4) for e in energies_qiskit]
energies_qutip = [round(e, 4) for e in energies_qutip]
#print(energies_qiskit)
#print(energies_qutip)
#print(tlist)

# Plot --------------------------------------------------------------------------------
plt.plot(tlist, energies_qutip, label='QuTiP (Exact)', linewidth=2)
plt.plot(tlist, energies_qiskit, '--', label='Qiskit (Trotterized)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Total Energy <H>')
plt.title('Energy: QuTiP vs Qiskit')
plt.legend()
plt.show()
