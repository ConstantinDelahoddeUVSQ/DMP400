# A présent nous vérifions la cohérence de nos valeurs pour les points de contact des faisceaux avec la plaque, 
# pour se faire nous calculons manuellement nos valeurs et les comparons a celles que l'on trouve avec le module deviation




import sys, os
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_partie_bleue = os.path.join(folder, "SIMS", "deviation_electrique", "Code")
sys.path.append(path_partie_bleue)


# --- Importations des modules de simulation ---
try:
    import deviation # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Impossible d'importer les modules de simulation.")
    print(f"Vérifiez l'existence des fichiers .py dans:")
    print(f"  '{path_partie_bleue}'")
    print("Assurez-vous que ces dossiers sont corrects et contiennent les fichiers __init__.py si nécessaire.")
    sys.exit(1)



v0 = 1e6    
theta_deg = 30  
theta = np.radians(theta_deg)  
y0 = 0.05       
E = -100.0      
x0 = 0.0       


m_sur_q = np.linspace(1, 100, 500)  # évite division par 0
# M_sur_q est juste la masse m car on va considerer que q = 1

xs_values = []

for ele in m_sur_q:
    m = ele *constants.u
    q = 1.0 * constants.e
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    discriminant = (v0 * cos_theta)**2 - (2 * y0 * q * E) / m
    
    if discriminant >= 0:
        numerator = v0 * cos_theta - np.sqrt(discriminant)
        denominator = (q / m) * E
        xs = (numerator / denominator) * v0 * sin_theta + x0
    else:
        xs = np.nan  # valeur non définie si racine carrée négative
    
    xs_values.append(xs)


xs_deviation = []

for i in range (len(m_sur_q)) :
    p =deviation.particule((m_sur_q[i],1) , v0 , theta, y0)
    xs_deviation.append(p.point_contact(E))


# Tracé
plt.figure(figsize=(10, 6))
plt.plot(m_sur_q, xs_values, label="$x_s$ en fonction de $m/q$")
plt.plot(m_sur_q, xs_deviation, label= "xs deviation")
plt.xlabel("$m/q$")
plt.ylabel("$x_s$")
plt.title("Variation de $x_s$ en fonction du rapport $m/q$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# On remarque que les tracés sont confondus. Cela confirme donc la correcte exécution de notre programme.