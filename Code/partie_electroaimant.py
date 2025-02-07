# Objectif 1

import matplotlib.pyplot as plt
import numpy as np

# Objet d'une particule décrite avec son rapport masse/charge et sa vitesse initiale
class particule :
    def __init__(self, rapport_masse_charge : float, v_initiale : float) :
        """
        Objet particule traversant un champ magnétique d'axe z

        Parameters
        ----------
        rapport_masse_charge : float
            Masse (en Kg) / Charge (en C) de la particule
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s)   
        """
        self.mq = rapport_masse_charge
        self.vo = v_initiale

    # Niveau 5 : L'equation de la trajectoire d'une particule en fonction de son rapport masse/charge et sa vitesse initiale
    def equation_trajectoire(self, x : float, Bz : float) -> float :    
        """
        La position y de la particule en x

        Parameters
        ----------
        x : float
            Position en x de la particule (en m)
        Bz : float
            Valeur du champ magnétique d'axe z (en T)

        Returns
        -------
        float
            Position en y de la particule (en m)
        """             
        prefix = self.mq / Bz
        return self.vo * prefix * np.sin(np.arccos(1 - x / (self.vo * prefix)))

    # Niveau 4 : Renvoie un tuple de la trajectoire de la particule (liste des abscisses, liste des ordonnées)
    def trajectoire(self, Bz : float, x_min : float, x_max : float, n_points : int = 10000) -> tuple[list[float], list[float]] :
        """
        Calcule la trajectoire entre un x minimum et un x maximum

        Parameters
        ----------
        Bz : float
            Valeur du champ magnétique d'axe z (en T)
        x_min : float
            Position en x minimale (en m)
        x_max : float
            Position en x maximale (en m)   
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max
        
        Returns
        -------
        tuple of (list of float, list of float)
            - Positions en x
            - Positions en y
        
        """
        x = np.linspace(x_min, x_max, n_points)
        return x, self.equation_trajectoire(x, Bz)
    
    # Niveau 3 : Trace la trajectoire de la particule dans le champ Bz avec matplotlib en 2d
    def tracer_trajectoire(self, ax, Bz : float, x_min : float, x_max : float, n_points : int = 10000) -> None : 
        """
        Trace la trajectoire entre x_min et x_max sur ax

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            L'axe matplotlib sur lequel on veut tracer la trajectoire
        Bz : float
            Valeur du champ magnétique d'axe z (en T)
        x_min : float
            Position en x minimale (en m)
        x_max : float
            Position en x maximale (en m)
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max

        """
        x, y = self.trajectoire(Bz, x_min, x_max, n_points)
        ax.plot(x, y, label=str(self.mq))



def tracer_ensemble_trajectoires(rapports_masse_charge_particules : list[float], vitesse_initiale : float, Bz : float, x_min : float, x_max : float) -> None:
    """
    Trace les trajectoires entre x_min et x_max pour un ensemble de particules d'un faisceau

    Parameters
    ----------
    rapports_masse_charge_particules : list of float
        Masse (en Kg) / Charge (en C)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale en y commune à toutes les particules du faisceau
    Bz : float
            Valeur du champ magnétique d'axe z (en T)
    x_min : float
            Position en x minimale (en m)
        x_max : float
            Position en x maximale (en m)
    """
    particules = [particule(rapport_masse_charge, vitesse_initiale) for rapport_masse_charge in rapports_masse_charge_particules]    # Liste d'objets particule représentant toutes les particules
    fig, ax = plt.subplots()
    for particule_locale in particules :
        particule_locale.tracer_trajectoire(ax, Bz, x_min, x_max)
    
    ax.legend()
    plt.show()

# Test du programme (valeurs non représentatives)
rapports_masse_charge = [1e-27/1.602e-19, 2e-27/1.602e-19, 3e-27/1.602e-19]
vitesse_initiale = 1
Bz = 10e-10
x_min, x_max = 0, 5

tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_min, x_max)

