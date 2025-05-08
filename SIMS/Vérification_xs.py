import sys, os
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.abspath(__file__))
path_partie_bleue = os.path.join(folder, "Partie Bleue (accélération)", "Code")
path_partie_verte = os.path.join(folder, "Partie Verte (déviation magnétique)", "Code")
sys.path.append(path_partie_bleue)
sys.path.append(path_partie_verte)

# --- Importations des modules de simulation ---
try:
    import deviation # type: ignore
    import partie_electroaimant # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Impossible d'importer les modules de simulation.")
    print(f"Vérifiez l'existence des fichiers .py dans:")
    print(f"  '{path_partie_bleue}'")
    print(f"  '{path_partie_verte}'")
    print("Assurez-vous que ces dossiers sont corrects et contiennent les fichiers __init__.py si nécessaire.")
    sys.exit(1)



v0 = 1e6    
theta_deg = 30  
theta = np.radians(theta_deg)  
y0 = 0.05       
E = -100.0      
x0 = 0.0       


m = np.linspace(1, 100, 500)  # évite division par 0
# on prend q =1 donc le rapport masse sur devient juste m

xs_values = []

for e in m:
    m = e *constants.u
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

for i in range (len(m)) :
    p =deviation.particule((m[i],1) , v0 , theta, y0)
    xs_deviation.append(p.point_contact(E))


print(xs_deviation)


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
