from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def expectation_zz(counts, shots):

    exp = 0

    for bitstring, count in counts.items():

        if bitstring in ['00', '11']:
            exp += count
        else:
            exp -= count

    return exp / shots


shots = 10000

# Target circuit -----------------------------------------------------

qc = QuantumCircuit(2, 2)

qc.h(0)
qc.cx(0, 1)

qc.measure([0,1], [0,1])
#print(qc)

# Ideal simulation (just shot noise) ---------------------------------

ideal_sim = AerSimulator()

compiled = transpile(qc, ideal_sim)

ideal_result = ideal_sim.run(
    compiled,
    shots=shots
).result()

ideal_counts = ideal_result.get_counts()

ideal_expval = expectation_zz(ideal_counts, shots)

print("Ideal <ZZ> =", ideal_expval)

# Add depolarizing noise ---------------------------------------------

p = 0.1

noise_model = NoiseModel()

cx_error = depolarizing_error(p, 2)

noise_model.add_all_qubit_quantum_error(
    cx_error,
    ['cx']
)

noisy_sim = AerSimulator(noise_model=noise_model)

compiled_noisy = transpile(qc, noisy_sim)

noisy_result = noisy_sim.run(
    compiled_noisy,
    shots=shots
).result()

noisy_counts = noisy_result.get_counts()

noisy_expval = expectation_zz(noisy_counts, shots)

print("Noisy <ZZ> =", noisy_expval)

# Estimation circuit -------------------------------------------------

# remove single-qubit gates
# keep only CNOT's

est_qc = QuantumCircuit(2, 2)

est_qc.cx(0, 1)

est_qc.measure([0,1], [0,1])

compiled_est = transpile(est_qc, noisy_sim)

est_result = noisy_sim.run(
    compiled_est,
    shots=shots
).result()

est_counts = est_result.get_counts()

est_expval = expectation_zz(est_counts, shots)

print("Estimation circuit <ZZ> =", est_expval)


# Estimate depolarizing parameter ------------------------------------

p_est = 1 - est_expval

print("Estimated p =", p_est)


# Mitigate -----------------------------------------------------------

mitigated_expval = noisy_expval / (1 - p_est)

print("Mitigated <ZZ> =", mitigated_expval)