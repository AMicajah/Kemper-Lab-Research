import numpy as np
from qutip import *
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt
# Parameters
N = 4
J = -1.0
hx = 0.5
hz = 0.3

# Matrix Diagonalization (qutip) ------------------------------------------------------------------------

def mfit_hamiltonian(N, J, hx, hz):
    
    # Pauli matrices
    sx = sigmax()
    sz = sigmaz()
    id2 = qeye(2)
    
    H = 0
    
    # ZZ interaction term
    for i in range(N - 1):  # nearest neighbors
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sz)
            elif j == i + 1:
                ops.append(sz)
            else:
                ops.append(id2)
        H += J * tensor(ops)
    
    # Transverse field term
    for i in range(N):
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sx)
            else:
                ops.append(id2)
        H += hx * tensor(ops)
    
    # Longitudinal field term
    for i in range(N):
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sz)
            else:
                ops.append(id2)
        H += hz * tensor(ops)
    
    return H

H = mfit_hamiltonian(N, J, hx, hz)

# Exact diagonalization
eigenvalues = H.eigenenergies()

ground_energy = eigenvalues[0]

print("Exact ground state energy:", ground_energy)

#VQE Implementation ------------------------------------------------------------------------------------------

def build_mfim_hamiltonian(N, J, hx, hz):
    paulis = []
    coeffs = []
    
    # ZZ nearest neighbor terms
    for i in range(N - 1):
        z_string = ['I'] * N
        z_string[i] = 'Z'
        z_string[i+1] = 'Z'
        paulis.append("".join(z_string))
        coeffs.append(J)
    
    # Transverse field terms 
    for i in range(N):
        x_string = ['I'] * N
        x_string[i] = 'X'
        paulis.append("".join(x_string))
        coeffs.append(hx)
    
    # Longitudinal field terms 
    for i in range(N):
        z_string = ['I'] * N
        z_string[i] = 'Z'
        paulis.append("".join(z_string))
        coeffs.append(hz)
    
    return SparsePauliOp(paulis, coeffs)

def build_hva_ansatz(N, layers):
    qc = QuantumCircuit(N)
    
    params = ParameterVector("theta", length=3*layers)
    #print(params.params[0].numeric())
    #for x in dir(params.params[0]):
        #print(x)
    for l in range(layers):
        alpha = params[3*l]
        beta  = params[3*l + 1]
        gamma = params[3*l + 2]
        
        # Brick-layer ZZ interactions
        # Even Pairs
        for i in range(0, N-1, 2):
            qc.rzz(alpha, i, i+1)
            
        # Odd Pairs
        for i in range(1,N-1,2):
            qc.rzz(alpha, i, i+1)
            
        # Z rotations
        for i in range(N):
            qc.rz(beta, i)
        
        # X rotations
        for i in range(N):
            qc.rx(gamma, i)
        print(qc)
    return qc, params

def run_vqe_fake_backend(N, layers, H, maxiter=100, shots=1024, n_starts = 5):

    """
    VQE noisy sim with a fake backend (FakeManilaV2 + AerSimulator),
    handling X terms by rotating them to the Z-basis for measurement.
    
    Run VQE with multiple random initializations to reduce local minima issues.
    Returns the best result and its energy history.
    
    """
    energy_history_best = None
    best_result = None
    lowest_energy = np.inf
    
    # Fake backend and Noise Model
    backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(backend)

    # Simulator
    sim = AerSimulator()
    sim.set_options(noise_model=noise_model,
                shots=shots)

    # HVA Ansantz
    qc, params = build_hva_ansatz(N, layers)

    def cost(theta_values):
        expval_total = 0.0
        qc_bound = qc.assign_parameters(dict(zip(params, theta_values)))
    
        for pauli, coeff in zip(H.paulis, H.coeffs):
            pauli_str = pauli.to_label()
            
            qc_term = QuantumCircuit(N,N)
            qc_term.compose(qc_bound, inplace=True)
        
            # Rotate X -> Z
            for i, p in enumerate(pauli_str):
                if p == 'X':
                    qc_term.h(i)
        
            qc_term.measure(range(N), range(N))
        
            # Run simulator
            job = sim.run(qc_term)
            result = job.result()
            counts = result.get_counts()
        
            term_exp = 0.0
            for bitstring, c in counts.items():
                z_vals = np.array([1 if b=='0' else -1 for b in bitstring[::-1]])
                term = 1.0
                for i, p in enumerate(pauli_str):
                    if p == 'I':
                        continue
                    term *= z_vals[i]
                term_exp += c * term
            term_exp /= sum(counts.values())
            expval_total += coeff * term_exp

        return expval_total.real

    # Multi-start loop
    for start in range(n_starts):
        initial_point = np.random.uniform(0, 2*np.pi, 3*layers)
        energy_history = []

        # Wrap cost to save history
        def cost_with_history(theta_values):
            e = cost(theta_values)
            energy_history.append(e)
            return e

        result = minimize(cost_with_history, initial_point, method="COBYLA",
                          options={"maxiter": maxiter})

        if result.fun < lowest_energy:
            lowest_energy = result.fun
            best_result = result
            energy_history_best = energy_history

    return best_result, energy_history_best

H_vqe = build_mfim_hamiltonian(N, J, hx, hz)

for p in [1, 2, 3, 4]:
    result, history = run_vqe_fake_backend(N, p, H_vqe, maxiter=200, n_starts = 5)
    print(f"Layers = {p}")
    print("VQE energy:", result.fun)
    print("Error:", result.fun - ground_energy)
    print()

    plt.plot(history, label = f"Layers={p}")
    
plt.axhline(ground_energy, linestyle='--', label="Exact")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.show()