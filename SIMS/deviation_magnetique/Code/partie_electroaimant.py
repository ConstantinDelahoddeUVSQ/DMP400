# Objectif 1

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.constants as constants
from scipy.optimize import fsolve

class particule :
    def __init__(self, masse_charge : tuple[float, float], v_initiale : float) -> None :
        """
        Objet particule traversant un champ magnétique B // z.
        Vitesse initiale supposée selon +y.

        Parameters
        ----------
        masse_charge : tuple (float, float)
            Masse (u), Charge (e).
        v_initiale : float
            Vitesse initiale en y (m/s)
        """
        mass_u, charge_e = masse_charge
        self.mq = (mass_u * constants.u) / (abs(charge_e) * constants.e) # (toujours positif)
        self.vo = v_initiale
        self.m = mass_u
        # Stocker la charge signée pour l'affichage
        self.charge_affichage = charge_e

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
        with np.errstate(invalid='ignore') :
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

    def tracer_trajectoire(self, ax, Bz : float, x_min : float, x_max : float, color=None, label=None, n_points : int = 10000) -> None:
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
        color : str or list or array
            couleur à tracer
        label : str
            label du tracé
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max

        """
        x, y = self.trajectoire(Bz, x_min, x_max, n_points)
        if len(x) == 0: return # Ne rien tracer si vide

        plot_kwargs = {'c': color if color else None}
        if label: plot_kwargs['label'] = label
        ax.plot(x, y, **plot_kwargs)


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
def tracer_ensemble_trajectoires(masses_charges_particules : list[tuple[float, float]], vitesse_initiale : float, Bz : float, x_detecteur : float, labels_particules: list[str] = None, create_plot : bool = True, ax = None) -> None:
    """
    Trace les trajectoires entre 0 et x_detecteur pour un ensemble de particules d'un faisceau

    Parameters
    ----------
    masses_charges_particules : list of tuple of float
        Liste des Masse (en unités atomiques), Charge (nombre de charges élémentaires)  pour les particules
    vitesse_initiale : float
        Vitesse intiale en y commune à toutes les particules du faisceau
    Bz : float
        Valeur du champ magnétique d'axe z (en T)
    x_detecteur : float
        L'abscisse du détecteur (en m)
    labels_particules : list of str 
        Liste des labels pour chaque particule
    create_plot : bool
        True s'il faut que la fonction crée un plot et l'affiche, False sinon (et l'argument ax est nécéssaire)
    ax : matplotlib.axes.Axes
        Axe matplotlib sur lequel le tracé sera fait (uniquement si create_plot = False)
    """
    particules = [particule(masse_charge, vitesse_initiale) for masse_charge in masses_charges_particules]    # Liste d'objets particule représentant toutes les particules
    if ax == None or create_plot == True :
        fig, ax = plt.subplots()
    
    all_y_contact = []
    labels = {}
    for i in range(len(particules)) :
        labels[particules[i]] = labels_particules[i] 

    for particule_locale in particules :
        y_contact = particule_locale.equation_trajectoire(x_detecteur, Bz)
        all_y_contact.append(y_contact)
        particule_locale.tracer_trajectoire(ax, Bz, 0, x_detecteur, label=labels[particule_locale])
    
    if np.all(np.isnan(all_y_contact)):
        all_y_contact = [0.07 * x_detecteur]
    ax.plot([x_detecteur, x_detecteur], [ax.get_ybound()[0], ax.get_ybound()[1]], c='black', linewidth=5, label='Détecteur')
    ax.set_xlabel('Position x (en m)')
    ax.set_ylabel('Position y (en m)')
    ax.set_title(f"Déviation magnétique dans un champ de {Bz:.3f} T")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    if create_plot :
        plt.show()


'''
Test de la fonction tracer_ensemble_trajectoires (valeurs non représentatives)
On trace les trajectoires de particules avec des (masses, charges) différentes dans un champ magnétique donné
'''
# if __name__ == '__main__' :
#     rapports_masse_charge = [(1, 1), (2, 1), (3, 1)]
#     vitesse_initiale = 1e7
#     Bz = 1
#     x_detecteur = 1e-4
    
#     tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_detecteur, labels_particules=['P1', 'P2', 'P3'])
