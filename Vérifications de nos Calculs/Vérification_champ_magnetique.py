import sys, os
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_partie_verte = os.path.join(folder, "SIMS", "deviation_magnetique", "Code")
sys.path.append(path_partie_verte)
import partie_electroaimant


m = 1*sc.u
q = 1*sc.e
m_sur_q = m / q  # Rapport masse / charge (kg/C)

# Autres paramètres
V_0 = 1e5  # Vitesse initiale (m/s)
B_z = 1e-3  # Champ magnétique (T)

# Constante utile
K = 1 / (V_0 * m_sur_q)

# Calcul de la borne max admissible pour x
x_max = 2 / K
x_values = np.linspace(0, x_max, 500)

# Définition de la fonction y(x)
def y(x):
    term = 1 - K * x
    term = np.clip(term, -1, 1)  # Sécurité numérique
    return (V_0 * m_sur_q) * np.sin(np.arccos(term))

# Calcul des y
y_values = y(x_values)

y_deviation = []
for i in range(len(x_values)) :
    p =  partie_electroaimant.particule((m,q) , V_0 )
    y_deviation.append(p.equation_trajectoire(i, B_z))

# Tracé
# plt.plot(x_values, y_values, label='y(x)')
plt.plot(x_values, y_deviation)
plt.xlabel('x (m)')
plt.ylabel('y(x) (m)')
plt.title('Trajectoire y(x) d\'une particule dans un champ Bz')
plt.grid(True)
plt.legend()
plt.show()




