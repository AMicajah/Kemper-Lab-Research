import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, SparsePauliOp

# Parameters
N = 2
Jx = 1.0
Jy = 0.5
Jz = 0.8

tlist = np.linspace(0, 10, 200)
dt = tlist[1] - tlist[0]

n_traj = 2000   # number of trajectories

# Heisenberg Hamiltonian
def build_heisenberg_hamiltonian(N, Jx, Jy, Jz):
    paulis = []
    coeffs = []
    
    for i in range(N - 1):
        for P, J in zip(['X','Y','Z'], [Jx, Jy, Jz]):
            s = ['I'] * N
            s[i] = P
            s[i+1] = P
            paulis.append("".join(s))
            coeffs.append(J)
    
    return SparsePauliOp(paulis, coeffs)

H = build_heisenberg_hamiltonian(N, Jx, Jy, Jz).to_matrix()

# Jump operators (sigma^-)
sigma_minus = np.array([[0, 0],
                        [1, 0]], dtype=complex)

I = np.eye(2)

L1 = np.kron(sigma_minus, I)
L2 = np.kron(I, 0.5 * sigma_minus)

jump_ops = [L1, L2]

# Precompute L^daggerL
LdL = [L.conj().T @ L for L in jump_ops]

# Effective Hamiltonian
H_eff = H - (1j/2) * sum(LdL)

# Observables
sz1 = SparsePauliOp(["ZI"], [1.0]).to_matrix()
sz2 = SparsePauliOp(["IZ"], [1.0]).to_matrix()

sz_exp1_avg = np.zeros(len(tlist))
sz_exp2_avg = np.zeros(len(tlist))

# Trajectory loop
for traj in range(n_traj):
    
    # Initial state |10>
    psi = Statevector(np.kron([0,1], [1,0])).data
    
    sz_exp1 = []
    sz_exp2 = []
    
    for t in tlist:
        
        # record observables
        sz_exp1.append(np.real(np.vdot(psi, sz1 @ psi)))
        sz_exp2.append(np.real(np.vdot(psi, sz2 @ psi)))
        
        # non-unitary evolution
        psi = (np.eye(4) - 1j * H_eff * dt) @ psi
        
        # jump probabilities
        probs = [dt * np.real(np.vdot(psi, LdL_k @ psi)) for LdL_k in LdL]
        p_total = sum(probs)
        
        # stochastic decision
        r = np.random.rand()
        
        if r < p_total:
            # choose which jump
            probs_norm = np.array(probs) / p_total
            k = np.random.choice(len(jump_ops), p=probs_norm)
            
            # apply jump
            psi = jump_ops[k] @ psi
        
        # renormalize
        psi = psi / np.linalg.norm(psi)
    
    # accumulate averages
    sz_exp1_avg += np.array(sz_exp1)
    sz_exp2_avg += np.array(sz_exp2)

# Average over trajectories
sz_exp1_avg /= n_traj
sz_exp2_avg /= n_traj

# Plot
plt.figure(figsize=(7,5))
plt.plot(tlist, sz_exp1_avg, 'b-', label='Site 1 <σ_z>')
plt.plot(tlist, sz_exp2_avg, 'r-', label='Site 2 <σ_z>')
plt.xlabel('Time')
plt.ylabel('<σ_z>')
plt.title('Two Site Dissipative Heisenberg Model')
plt.legend()
plt.tight_layout()
plt.show()