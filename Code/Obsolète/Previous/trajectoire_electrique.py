# Tracer la trajectoire avec le champ électrique quelconque
import numpy as np
import matplotlib.pyplot as plt

m = 1e-30
q = 1.6e-19
Ex, Ey, Ez = 1e-12, 0.5e-12, -0.5e-12
vx0, vy0, vz0 = 0, 1, 0

def tracer_avec_temps_2d() :
    t = np.linspace(0, 7, 1000)

    x = lambda t : (q / (2 * m)) * Ex * t*t + vx0 * t
    y = lambda t : (q / (2 * m)) * Ey * t*t + vy0 * t

    xt = x(t)
    yt = y(t)

    plt.plot(xt, yt)
    plt.show()

# tracer_avec_temps_2d()


def tracer_avec_temps_3d() :
    t = np.linspace(0, 7, 1000)

    x = lambda t : (q / (2 * m)) * Ex * t*t + vx0 * t
    y = lambda t : (q / (2 * m)) * Ey * t*t + vy0 * t
    z = lambda t : (q / (2 * m)) * Ez * t*t + vz0 * t

    xt = x(t)
    yt = y(t)
    zt = z(t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xt, yt, zt)
    plt.show()

tracer_avec_temps_3d()