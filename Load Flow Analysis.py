# Load-Flow-Analysis
import numpy as np

# ---------------------------------------------------------
# LOAD FLOW ANALYSIS USING GAUSS-SEIDEL METHOD
# ---------------------------------------------------------

# Number of buses
nbus = 3

# Bus types:
# 1 = Slack bus
# 2 = PV bus
# 3 = PQ bus
bus_type = np.array([1, 2, 3])

# Specified generation and load (p.u.)
# P = MW, Q = MVAR in per-unit form
P_gen = np.array([0.0, 0.50, 0.00])
Q_gen = np.array([0.0, 0.00, 0.00])

P_load = np.array([0.0, 0.20, 0.60])
Q_load = np.array([0.0, 0.10, 0.30])

# Net specified power
P_spec = P_gen - P_load
Q_spec = Q_gen - Q_load

# ---------------------------------------------------------
# Y-BUS MATRIX
# ---------------------------------------------------------

Ybus = np.array([
    [10-20j, -5+10j, -5+10j],
    [-5+10j, 8-16j, -3+6j],
    [-5+10j, -3+6j, 8-16j]
], dtype=complex)

# ---------------------------------------------------------
# INITIAL VOLTAGES
# ---------------------------------------------------------

V = np.ones(nbus, dtype=complex)

# Slack bus voltage
V[0] = 1.05 + 0j

# PV bus voltage magnitude
V[1] = 1.01 + 0j

# PQ bus initial voltage
V[2] = 1.0 + 0j

# Tolerance and maximum iterations
tolerance = 1e-6
max_iterations = 100

# ---------------------------------------------------------
# GAUSS-SEIDEL ITERATION
# ---------------------------------------------------------

for iteration in range(max_iterations):

    V_old = V.copy()

    for i in range(nbus):

        # Skip slack bus
        if bus_type[i] == 1:
            continue

        # Calculate sum of Yij * Vj
        sum_yv = 0 + 0j

        for j in range(nbus):
            if j != i:
                sum_yv += Ybus[i, j] * V[j]

        # -------------------------------------------------
        # PQ BUS
        # -------------------------------------------------
        if bus_type[i] == 3:

            S = P_spec[i] + 1j * Q_spec[i]

            V[i] = (1 / Ybus[i, i]) * (
                np.conj(S / V[i]) - sum_yv
            )

        # -------------------------------------------------
        # PV BUS
        # -------------------------------------------------
        elif bus_type[i] == 2:

            # Calculate reactive power at PV bus
            I = sum(Ybus[i, j] * V[j] for j in range(nbus))
            S = V[i] * np.conj(I)

            Q_i = S.imag

            S = P_spec[i] + 1j * Q_i

            V_temp = (1 / Ybus[i, i]) * (
                np.conj(S / V[i]) - sum_yv
            )

            # Maintain specified voltage magnitude
            V[i] = abs(V[i]) * V_temp / abs(V_temp)

    # Check convergence
    error = max(abs(V - V_old))

    if error < tolerance:
        break

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n========================================")
print("       LOAD FLOW ANALYSIS RESULTS")
print("========================================")

print(f"\nNumber of iterations = {iteration + 1}")
print(f"Maximum error = {error:.8f}")

print("\nBus Voltages:")
print("----------------------------------------")
print("Bus\tMagnitude(p.u.)\tAngle(deg)")
print("----------------------------------------")

for i in range(nbus):
    magnitude = abs(V[i])
    angle = np.angle(V[i], deg=True)

    print(f"{i+1}\t{magnitude:.6f}\t\t{angle:.6f}")

# ---------------------------------------------------------
# CALCULATE BUS POWER
# ---------------------------------------------------------

print("\nBus Power:")
print("----------------------------------------")
print("Bus\tP (p.u.)\tQ (p.u.)")
print("----------------------------------------")

for i in range(nbus):

    I = sum(Ybus[i, j] * V[j] for j in range(nbus))
    S = V[i] * np.conj(I)

    print(f"{i+1}\t{S.real:.6f}\t{S.imag:.6f}")

print("\n========================================")
print("          ANALYSIS COMPLETED")
print("========================================")
