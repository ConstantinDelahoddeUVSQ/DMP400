import numpy as np
import matplotlib.pyplot as plt

v0 = 10.0      
theta_deg = 45  
theta = np.radians(theta_deg)  
y0 = 0.05       
E = 100.0      
x0 = 0.0       


m_sur_q = np.linspace(0.01, 10, 500)  # évite division par 0

xs_values = []

for ratio in m_sur_q:
    m = ratio
    q = 1.0  
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

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(m_sur_q, xs_values, label="$x_s$ en fonction de $m/q$")
plt.xlabel("$m/q$")
plt.ylabel("$x_s$")
plt.title("Variation de $x_s$ en fonction du rapport $m/q$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
