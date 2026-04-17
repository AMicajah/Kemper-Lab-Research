import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Jx = 1.0
Jy = 0.5
Jz = 0.8

tlist = np.linspace(0, 10, 200)

# Pauli operators for each site
sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
sy1 = qt.tensor(qt.sigmay(), qt.qeye(2))
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))

sx2 = qt.tensor(qt.qeye(2), qt.sigmax())
sy2 = qt.tensor(qt.qeye(2), qt.sigmay())
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

# Hamiltonian: H = J_xXX + J_yYY + J_zZZ
H = Jx*sx1*sx2 + Jy*sy1*sy2 + Jz*sz1*sz2
#print(H)
# Jump operators
L1 = qt.sigmam()           
L2 = 0.5 * qt.sigmam()     
c_ops = [qt.tensor(L1, qt.qeye(2)), qt.tensor(qt.qeye(2), L2)]
#print(c_ops)
# Initial state (asymmetric)
psi0 = qt.tensor(qt.basis(2,1), qt.basis(2,0))
rho0 = qt.ket2dm(psi0)


L = qt.liouvillian(H, c_ops)
result_L = qt.mesolve(L, rho0, tlist)

# Expectation values for each site
sz_exp1 = [qt.expect(sz1, state) for state in result_L.states]
sz_exp2 = [qt.expect(sz2, state) for state in result_L.states]


plt.figure(figsize=(7,5))
plt.plot(tlist, sz_exp1, 'b-', label='Site 1 <σ_z>')
plt.plot(tlist, sz_exp2, 'r-', label='Site 2 <σ_z>')
plt.xlabel('Time')
plt.ylabel('<σ_z>')
plt.title('Two-site Heisenberg')
plt.legend()
plt.tight_layout()
plt.show()