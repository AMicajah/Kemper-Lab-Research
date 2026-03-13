from qutip import *

# Pauli matrices
sx = sigmax()
sz = sigmaz()
id2 = qeye(2)
    
def mfit_hamiltonian(N, J, hx, hz):
    
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

def reconstruct_coefficients(H, N):

    dim = 2**N

    print("Recovered coefficients: ")

    # X terms
    for i in range(N):
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sx)
            else:
                ops.append(id2)
        Pi = tensor(ops)

        ci = (Pi * H).tr() / dim
        print(f"X{i}: {ci}")

    # Z terms
    for i in range(N):
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sz)
            else:
                ops.append(id2)
        Pi = tensor(ops)

        ci = (Pi * H).tr() / dim
        print(f"Z{i}: {ci}")

    # ZZ terms
    for i in range(N-1):
        ops = []
        for j in range(N):
            if j == i or j == i+1:
                ops.append(sz)
            else:
                ops.append(id2)
        Pi = tensor(ops)

        ci = (Pi * H).tr() / dim
        print(f"Z{i}Z{i+1}: {ci}")

N = 3
J = 0.7
hx = 0.5
hz = 0.2

H = mfit_hamiltonian(N, J, hx, hz)

reconstruct_coefficients(H, N)