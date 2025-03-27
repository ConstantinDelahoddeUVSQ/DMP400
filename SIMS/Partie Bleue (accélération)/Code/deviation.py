# Objectif 2

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.constants as constants


def calcul_champ_electrique(charge_plaque : float, surface : float) -> float :
    """
    Fonction calculant le champ électrique uniforme créé par l'échantillon (plaque chargée)

    Parameters
    ----------
    charge_plaque : float
        Charge totale de la plaque (en C)
    surface : float
        Surface totale de la plaque (en m²)
    
    Returns
    -------
    float
        Valeur du champ électrique à proximité de la plaque dirigé selon y
    """
    if surface > 0 :
        return charge_plaque / (2 * surface * constants.epsilon_0)
    else : raise ValueError("La surface ne peut être nulle ou négative")


def champ_electrique_v2(distance : float, différence_potentiel : float) -> float :
    return différence_potentiel / distance


class particule :
    def __init__(self, masse_charge : tuple[int, int], v_initiale : float = 0, angle_initial : float = np.pi / 4, hauteur_initiale : float = 0.5) -> None :
        """
        Objet particule avec vitesse initiale dévié par un champ électrique d'axe y

        Parameters
        ----------
        masse_charge : tuple of int
            Masse (en u) / Charge (en eV) de la particule
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s)   
        angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
        hauteur_initiale : float
            Coordonnée en y du point de départ
        """
        self.mq = masse_charge[0] * constants.u / masse_charge[1] / constants.eV
        self.vo = v_initiale
        self.angle = angle_initial
        self.height = hauteur_initiale
        self.m = masse_charge[0]
        self.c = masse_charge[1]

    def equation_trajectoire(self, x : float, E : float) -> float:
        """
        Equation de la trajectoire de la particule y(x)
        
        Parameters
        ----------
        x : float
            L'abscisse à laquelle on veut calculer la coordonnée en y
        E : float
            Valeur du champ électrique à proximité de la plaque dirigé selon y
        
        Returns
        -------
        float
            Coordonée y de la particule au point x
        """
        X = x / (self.vo * np.sin(self.angle))
        return 0.5 * E / self.mq * X * X - self.vo * np.cos(self.angle) * X + self.height

    def trajectoire(self, E : float, x_min : float, x_max : float, n_points : int = 10000) -> tuple[np.ndarray, np.ndarray] :
        """
        Calcule la trajectoire entre un x minimum et un x maximum

        Parameters
        ----------
        E : float
            Valeur du champ électrique à proximité de la plaque dirigé selon y
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
        return x, self.equation_trajectoire(x, E)
    
    def tracer_trajectoire(self, ax, E : float, x_min : float, x_max : float, n_points : int = 10000) -> None : 
        """
        Trace la trajectoire entre x_min et x_max sur ax

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            L'axe matplotlib sur lequel on veut tracer la trajectoire
        E : float
            Valeur du champ électrique à proximité de la plaque dirigé selon y
        x_min : float
            Position en x minimale (en m)
        x_max : float
            Position en x maximale (en m)
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max
        """
        x, y = self.trajectoire(E, x_min, x_max, n_points)
        ax.plot(x, y, label=f"Trajectoire de {self.m}u, {self.c}eV")
    
    
    def point_contact(self, E : float) -> float :
        """
        Calcule l'abscisse où la particule touche la plaque chargée

        Parameters
        ----------
        E : float
            Valeur du champ électrique à proximité de la plaque dirigé selon y
        
        Returns
        -------
        float
            abscisse du point de contact (depuis son abscisse initiale)
        """
        if (self.vo * np.cos(self.angle)) ** 2 - 2 * self.height * E / self.mq >= 0 :
            return self.mq * self.vo * np.sin(self.angle) / E * (self.vo * np.cos(self.angle) - np.sqrt((self.vo * np.cos(self.angle)) ** 2 - 2 * self.height * E / self.mq))
        else :
            raise ValueError("La particule n'a aucun point de contact avec l'échantillon")
        
    
    def angle_incident(self, E : float) -> float :
        """
        Calcule l'angle que la trajectoire forme avec l'axe y au point de contact avec la plaque en radians

        Parameters
        ----------
        E : float
            Valeur du champ électrique à proximité de la plaque dirigé selon y
        
        Returns
        -------
        float
            Angle formé par la trajectoire et l'axe y au point de contact avec l'échantillon en radians
        """
        E = calcul_champ_electrique(charge_plaque, surface)
        x_contact = self.point_contact(E)
        return  np.arctan(-1 / (E * x_contact / (self.mq * self.vo * np.sin(self.angle) * self.vo * np.sin(self.angle)) - 1 / np.tan(self.angle)))
    



def tracer_ensemble_trajectoires(masse_charge_particules : list[tuple[int, int]], vitesse_initiale : float, surface : float, charge_plaque : float, angle_initial=np.pi/6, hauteur_initiale = 0.15) -> None :
    """
    Trace les trajectoires entre jusqu'au contact de différentes particules

    Parameters
    ----------
    masse_charge_particules : list of tupleof int
        Masse (en unités atomiques), Charge (en eV)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale en y commune à toutes les particules du faisceau
    surface : float
        Surface totale de la plaque (en m²)
    charge_plaque : float
        Charge totale de la plaque (en C)
    angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ

    """
    particules = [particule(masse_charge, vitesse_initiale, angle_initial, hauteur_initiale) for masse_charge in masse_charge_particules]
    # E = calcul_champ_electrique(charge_plaque, surface)
    E = champ_electrique_v2(0.15, -5000)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_x_max = []
    for p in particules :
        x_max = p.point_contact(E)
        all_x_max.append(x_max)
        
        p.tracer_trajectoire(ax, E, 0, x_max)
        angle_incident = p.angle_incident(E)
        print(angle_incident, angle_incident * 180 / np.pi)

    ax.plot([0, max(all_x_max) * 1.2], [0, 0], c='black', linewidth=5, label='Echantillon')

    zoom_target_x, zoom_target_y = (min(all_x_max) + max(all_x_max)) * 0.5, 0
    zoom_factor_x, zoom_factor_y = (min(all_x_max) + max(all_x_max)) * 0.5, hauteur_initiale * 1.1
    ax.set_xlim(zoom_target_x - zoom_factor_x * 1.05, zoom_target_x + zoom_factor_x * 0.3)
    ax.set_ylim(zoom_target_y - zoom_factor_y * 0.05, zoom_target_y + zoom_factor_y)

    ax_zoom = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(ax_zoom, 'Zoom', 1, 2, valinit=1, valstep= 0.01)

    def update(val):
        zoom_factor = 10**(10 * ((1 / slider.val) - 1))
        ax.set_xlim(zoom_target_x - zoom_factor * zoom_factor_x * 1.05, zoom_target_x + zoom_factor * zoom_factor_x * 0.3)
        ax.set_ylim(zoom_target_y - zoom_factor * zoom_factor_y * 0.05, zoom_target_y + zoom_factor * zoom_factor_y)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.legend()
    plt.show()


if __name__ == '__main__' :
    rapports_mq, vo = [(1, 1), (2, 1), (3, 1)], 1e8
    charge_plaque, surface = -1e-4, 0.01

    tracer_ensemble_trajectoires(rapports_mq, vo, surface, charge_plaque)
