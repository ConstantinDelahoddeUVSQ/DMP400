# Objectif 1

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve 
import scipy.constants as constants


# Objet d'une particule décrite avec son rapport masse/charge et sa vitesse initiale
class particule :
    def __init__(self, masse_charge : tuple[int, int], v_initiale : float) -> None :
        """
        Objet particule traversant un champ magnétique d'axe z

        Parameters
        ----------
        masse_charge : tuple of int
            Masse (en uintés atomiques), Charge (en eV) de la particule
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s)   
        """
        self.mq = masse_charge[0] * constants.u / masse_charge[1] / constants.eV
        self.vo = v_initiale
        self.m = masse_charge[0]
        self.c = masse_charge[1]


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
    def trajectoire(self, Bz : float, x_min : float, x_max : float, n_points : int = 10000) -> tuple[np.ndarray, np.ndarray] :
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
        tuple of (numpy.ndarray, numpy.ndarray)
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
        ax.plot(x, y, label=f'{self.m}u, {self.c}eV')
    

    # Niveau 2.1 : Détermine la puissance du champ magnétique nécéssaire pour dévier une particule à un point précis
    def determiner_champ_magnetique(self, x_objective : float, y_objective : float, B0 : float = None) -> float :
        """
        Donne le champ magnétique pour dévier la particule en (x_objective, y_objective) depuis l'origine

        Parameters
        ----------
        x_objective : float
            Position en x voulue à l'état final
        y_objective : float
            Position en y voulue à l'état final
        B0 : float
            Valeur de départ de recherche du champ magnétique (pour la fonction fsolve de scipy)

        Returns
        -------
        float
            Champ magnétique (en T)
        """
        if B0 == None : B0 = self.mq
        equation_func = lambda B : y_objective - (self.mq * self.vo / B) * np.sin(np.arccos(1 - x_objective * B / (self.vo * self.mq)))
        return fsolve(equation_func, B0)[0]
        


# Niveau 2.2 : Tracer l'ensemble des trajectoires des particules d'un faisceau
def tracer_ensemble_trajectoires(masses_charges_particules : list[tuple[int, int]], vitesse_initiale : float, Bz : float, x_detecteur : float) -> None:
    """
    Trace les trajectoires entre x_min et x_max pour un ensemble de particules d'un faisceau

    Parameters
    ----------
    masses_charges_particules : list of tuple of int
        Masse (en unités atomiques), Charge (en eV)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale en y commune à toutes les particules du faisceau
    Bz : float
            Valeur du champ magnétique d'axe z (en T)
    x_detecteur : float
        L'abscisse du détecteur (en m)
    """
    particules = [particule(masse_charge, vitesse_initiale) for masse_charge in masses_charges_particules]    # Liste d'objets particule représentant toutes les particules
    fig, ax = plt.subplots()
    
    all_y_contact = []
    for particule_locale in particules :
        particule_locale.tracer_trajectoire(ax, Bz, 0, x_detecteur)
        all_y_contact.append(particule_locale.equation_trajectoire(x_detecteur, Bz))

    ax.plot([x_detecteur, x_detecteur], [min(all_y_contact) * 0.8, max(all_y_contact) * 1.1], c='black', linewidth=5, label='Détecteur')

    ax.legend()
    plt.show()



'''
Test de la fonction tracer_ensemble_trajectoires (valeurs non représentatives)

On trace les trajectoires de particules avec des rapports m/q différents dans un champ magnétique donné
'''
if __name__ == '__main__' :
    rapports_masse_charge = [(1, 1), (2, 1), (3, 1)]
    vitesse_initiale = 1
    Bz = 1.25e-08
    x_detecteur = 0.2
    
    tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_detecteur)


'''
Test de la fonction déterminer_champ_magnétique (valeurs non représentatives)

On cherche le champ magnétique pour dévier la trajectoire en x_max, x_max
Puis on trace la trajectoire jusqu'en x_max
On remarque que la particule finit effectivement à la position prévue
'''
# if __name__ == '__main__' :
#     rapports_masse_charge, vi = [1e-27/1.602e-19], 1
#     p = particule(rapports_masse_charge[0], vi)
#     x_min, x_max = 0, 0.5
#     Bz = p.determiner_champ_magnetique(x_max, x_max)

#     tracer_ensemble_trajectoires(rapports_masse_charge, vi, Bz, x_min, x_max)
    
