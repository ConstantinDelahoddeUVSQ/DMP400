import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

def champ_electrique_v2(distance: float, delta_V: float) -> float:
    return delta_V / distance

def derivees_partielles(v0, theta, y0, q_si, m_si, E):
    A = v0 * np.cos(theta)
    B = 2 * y0 * q_si * E / m_si
    C = q_si * E / m_si
    D = np.sqrt(np.maximum(A**2 - B, 0))
    dxs_dv0    = (np.sin(theta)/C)*(A-D) + (v0*np.sin(theta)*np.cos(theta)/C)*(1 - A/D)
    dxs_dtheta = (v0*np.cos(theta)/C)*(A-D) - (v0**2 * np.sin(theta)**2/C)*(1 - A/D)
    dxs_dy0    = -v0 * np.sin(theta) * q_si * E / (C * m_si * D)
    dxs_dq     = -(A-D)*E*v0*np.sin(theta)/(C**2*m_si) - v0*np.sin(theta)*y0*E/(C*m_si*D)
    dxs_dm     = (A-D)*q_si*E*v0*np.sin(theta)/(C**2*m_si**2) + v0*np.sin(theta)*y0*q_si*E/(C*m_si**2*D)
    dxs_dE     = -(A-D)*q_si*v0*np.sin(theta)/(C**2*m_si) - v0*np.sin(theta)*y0*q_si/(C*m_si*D)
    return dxs_dv0, dxs_dtheta, dxs_dy0, dxs_dq, dxs_dm, dxs_dE

def calculer_incertitude(v0, theta, y0, q_si, m_si, E, deltas):
    dx = derivees_partielles(v0, theta, y0, q_si, m_si, E)
    return np.sqrt(sum((d*Δ)**2 for d, Δ in zip(dx, deltas)))

# Paramètres communs
v0        = 1e8       # m/s
theta     = np.pi/6
y0        = 0.15      # m
distance  = 0.15      # m
delta_V   = -5000     # V

# On génère 1 000 masses entre 1 et 10 u, toutes charges = 1 eV
m_u_list = np.linspace(1, 10, 1_000)
q_e_list = np.ones_like(m_u_list)

E = champ_electrique_v2(distance, delta_V)

mq_vals = np.empty_like(m_u_list)
inc_vals = np.empty_like(m_u_list)

for i, (m_u, q_e) in enumerate(zip(m_u_list, q_e_list)):
    m_si = m_u * constants.u
    q_si = q_e * constants.e
    mq_vals[i] = m_si / q_si
    deltas = (
        v0*0.01,        # Δv0 = 1%
        theta*0.02,     # Δθ = 2%
        y0*0.05,        # Δy0 = 5%
        q_si*0.001,     # Δq = 0.1%
        m_si*0.001,     # Δm = 0.1%
        abs(E)*0.03     # ΔE = 3%
    )
    inc_vals[i] = calculer_incertitude(v0, theta, y0, q_si, m_si, E, deltas)

# Tracé final
plt.figure(figsize=(8,5))
plt.plot(mq_vals, inc_vals)
plt.xscale('log')
plt.xlabel("m/q (kg/C)")
plt.ylabel("Δ xs (m)")
plt.title("Incertitude sur xs avec 1 000 points (1–10 u)")
plt.show()
