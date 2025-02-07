# Objectif 1

import matplotlib.pyplot as plt
import numpy as np

# Objet d'une particule décrite avec son rapport masse/charge et sa vitesse initiale
class particule :
    def __init__(self, rapport_masse_charge : float, v_initiale : float) :
        self.mq = rapport_masse_charge
        self.vo = v_initiale

    # Niveau 5 : L'equation de la trajectoire d'une particule en fonction de son rapport masse/charge et sa vitesse initiale
    def equation_trajectoire(self, x : float, Bz : float) :                 
        prefix = self.mq / Bz
        return self.vo * prefix * np.sin(np.arccos(1 - x / (self.vo * prefix)))

    # Niveau 4 : Renvoie un tuple de la trajectoire de la particule (liste des abscisses, liste des ordonnées)
    def trajectoire(self, Bz : float, x_min : float, x_max : float, n_points=10000) :
        x = np.linspace(x_min, x_max, n_points)
        return (x, self.equation_trajectoire(x, Bz))
    
    # Niveau 3 : Trace la trajectoire de la particule dans le champ Bz avec matplotlib en 2d
    def tracer_trajectoire(self, ax,Bz : float, x_min : float, x_max : float, n_points=10000) :    # ax : la figure matplotlib pour plot la trajectoire ; Bz : champ magnétique 
        x, y = self.trajectoire(Bz, x_min, x_max, n_points)
        ax.plot(x, self.equation_trajectoire(x, Bz), label=str(self.mq))

    def trajectoire_fct_temps(self, Bz : float, t_max : int) :
        prefix = self.mq / Bz

        t = np.linspace(0, t_max, 10000)

        x = lambda t : - self.vo * prefix * np.cos(t / prefix)  + self.vo * prefix
        y = lambda t : self.vo * prefix * np.sin(t / prefix)

        xt = x(t)
        yt = y(t)

        plt.plot(xt, yt)
        plt.show()

    
    def position_sur_capteur(self, Bz : float, x_capteur : float) :
        prefix = self.m /(self.q * Bz)
        y = lambda x : self.vo * prefix * np.sin(np.arccos(1 - x / (self.vo * prefix)))
        return y(x_capteur)



def tracer_ensemble_trajectoires(rapports_masse_charge_particules : list, vitesse_initiale : float, Bz : float, x_min : float, x_max : float) :     # rapports_masse_charge_particules : liste des rapports masse/charge de toutes les particules du faisceau  ; vitesse_initiale : la vitesse initiale de toutes les particules du faisceau
    particules = [particule(rapport_masse_charge, vitesse_initiale) for rapport_masse_charge in rapports_masse_charge_particules]    # Liste d'objets particule de toutes les particules
    fig, ax = plt.subplots()
    for particule_locale in particules :
        particule_locale.tracer_trajectoire(ax, Bz, x_min, x_max)
    
    ax.legend()
    plt.show()

rapports_masse_charge = [1e-27/1.602e-19, 2e-27/1.602e-19, 3e-27/1.602e-19]
vitesse_initiale = 1
Bz = 10e-10
x_min, x_max = 0, 5

tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_min, x_max)



