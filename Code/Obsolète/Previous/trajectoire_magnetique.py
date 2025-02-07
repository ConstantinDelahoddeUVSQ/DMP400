# Tracer la trajectoire avec le champ magnétique axé selon z
import numpy as np
import matplotlib.pyplot as plt

m = 1e-30
q = 1.6e-19
Bz = -5.69e-12
k1 = 10
prefix = m /(q * Bz)

def tracer_avec_temps() :
    t = np.linspace(0, 2, 1000)

    x = lambda t : - k1 * prefix * np.cos(t / prefix)  + k1 * prefix
    y = lambda t : k1 * prefix * np.sin(t / prefix)

    xt = x(t)
    yt = y(t)

    plt.plot(xt, yt)
    plt.show()

tracer_avec_temps()

# tracer_avec_traj_archi()

def tracer_avec_traj_c() :
    y = lambda x : k1 * prefix * np.sin(np.arccos(1 - x / (k1 * prefix)))
    x = np.linspace(-15, 5, 10000)
    plt.plot(x, -y(x))
    plt.show()

# tracer_avec_traj_c()
