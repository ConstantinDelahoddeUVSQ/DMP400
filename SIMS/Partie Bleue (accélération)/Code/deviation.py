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


def champ_electrique_v2(distance: float, différence_potentiel: float) -> float:
    """
    Calcule le champ électrique uniforme entre deux plaques parallèles.

    Parameters
    ----------
    distance : float
        Distance entre les plaques (en m).
    difference_potentiel : float
        Différence de potentiel entre les plaques (en V).

    Returns
    -------
    float
        Intensité du champ électrique E = V/d (en V/m).

    Raises
    ------
    ValueError
        Si la distance est nulle ou négative.
    """
    if distance <= 0:
        raise ValueError("La distance doit être positive.")
    return différence_potentiel / distance


class particule :
    def __init__(self, masse_charge : tuple[int, int], v_initiale : float = 0, angle_initial : float = np.pi / 4, hauteur_initiale : float = 0.5) -> None :
        """
        Objet particule avec vitesse initiale dévié par un champ électrique d'axe y

        Parameters
        ----------
        masse_charge : tuple of int
            Masse (en u) / Charge (nombre de charge élémentaire) de la particule
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s)   
        angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
        hauteur_initiale : float
            Coordonnée en y du point de départ
        """
        if masse_charge[1] == 0:
            raise ValueError("La charge de la particule ne peut pas être nulle.")
        self.mq = masse_charge[0] * constants.u / masse_charge[1] / constants.e
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
        ax.plot(x, y, label=f"Trajectoire de {self.m}u, {self.c}e")
    
    
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
        with np.errstate(invalid='ignore') :
            if (self.vo * np.cos(self.angle)) ** 2 - 2 * self.height * E / self.mq >= 0 :
                if E != 0 :
                    return self.mq * self.vo * np.sin(self.angle) / E * (self.vo * np.cos(self.angle) - np.sqrt((self.vo * np.cos(self.angle)) ** 2 - 2 * self.height * E / self.mq))
                else :
                    return self.height / (self.vo * np.cos(self.angle))
            else :
                return None
                # raise ValueError("La particule n'a aucun point de contact avec l'échantillon")
        
    
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
        x_contact = self.point_contact(E)
        return  np.arctan(-1 / (E * x_contact / (self.mq * self.vo * np.sin(self.angle) * self.vo * np.sin(self.angle)) - 1 / np.tan(self.angle)))
    


def tracer_ensemble_trajectoires(masse_charge_particules : list[tuple[int, int]], vitesse_initiale : float, potentiel : float = 5000, angle_initial=np.pi/6, hauteur_initiale = 0.15, create_plot=True, ax=None) -> None :
    """
    Trace les trajectoires entre jusqu'au contact de différentes particules de manière statique

    Parameters
    ----------
    masse_charge_particules : list of tupleof int
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour toutes les particules
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
    particules_init = masse_charge_particules
    
    if create_plot or ax == None : 
        fig, ax = plt.subplots(figsize=(8, 4))

    E = champ_electrique_v2(hauteur_initiale, potentiel)

    particules = [particule(mq, vitesse_initiale, angle_initial, hauteur_initiale) for mq in particules_init]
    all_x_max = []
    non_contact_particules = []
    texte_angles = "Angles incidents :\n"
    is_contact = False

    for p in particules:
        if p.point_contact(E) is not None:
            x_max = p.point_contact(E)
            all_x_max.append(x_max)
            p.tracer_trajectoire(ax, E, 0, x_max)

            angle_incident = p.angle_incident(E)
            angle_deg = np.degrees(angle_incident)
            texte_angles += f"- {p.m}u, {p.c}e : {angle_deg:.2f}°\n"
            is_contact = True
        else : 
            non_contact_particules.append(p)
    
    if is_contact :
        ax.set_xlim(0, max(all_x_max) * 1.2)
    else :
        ax.set_xlim(0, hauteur_initiale)

    for p in non_contact_particules : 
        local_x_max = ax.get_xlim()[1]
        p.tracer_trajectoire(ax, E, 0, local_x_max * 1.2)
        all_x_max.append(local_x_max)
    
    if len(all_x_max) > 0:
        ax.plot([0, max(all_x_max) * 1.2], [0, 0], c='black', linewidth=5, label='Échantillon')
        ax.text(0.8, 0.5, texte_angles, transform=ax.transAxes,
            fontsize=10,bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.legend()

    if create_plot :
        plt.show()
    return ax


def tracer_ensemble_trajectoires_dynamique(masse_charge_particules : list[tuple[int, int]], vitesse_initiale : float, potentiel_min : float = -5000, potentiel_max : float = 5000, angle_initial=np.pi/6, hauteur_initiale = 0.15) -> None :
    """
    Trace les trajectoires entre jusqu'au contact de différentes particules dynamiquement (sliders)

    Parameters
    ----------
    masse_charge_particules : list of tupleof int
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour toutes les particules
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
    particules_init = masse_charge_particules

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  

    ax_zoom = plt.axes([0.1, 0.05, 0.8, 0.03])
    ax_E = plt.axes([0.1, 0.01, 0.8, 0.03])

    slider_zoom = Slider(ax_zoom, 'Zoom', 1, 2, valinit=1, valstep=0.01)
    slider_E = Slider(ax_E, 'Potentiel (V)', potentiel_min, potentiel_max, valinit=abs(champ_electrique_v2(hauteur_initiale, (potentiel_max - potentiel_min) / 2)), valstep=potentiel_max / 1000)

    def tracer(pot_val : float, zoom_val : float) -> None:
        """
        Fonction interne pour effacer et redessiner le graphique en fonction des valeurs actuelles.

        Parameters
        ----------
        pot_val : float
            Valeur actuelle du potentiel (V) issue du slider.
        zoom_val : float
            Valeur actuelle du facteur de zoom issue du slider.
        """
        ax.clear()

        particules = [particule(mq, vitesse_initiale, angle_initial, hauteur_initiale) for mq in particules_init]
        all_x_max = []
        texte_angles = "Angles incidents :\n"
        E_val = champ_electrique_v2(hauteur_initiale, pot_val)
        non_contact_particules = []
        for p in particules:
            if p.point_contact(E_val) is not None:
                x_max = p.point_contact(E_val)
                all_x_max.append(x_max)
                p.tracer_trajectoire(ax, E_val, 0, x_max)

                angle_incident = p.angle_incident(E_val)
                angle_deg = np.degrees(angle_incident)
                texte_angles += f"- {p.m}u, {p.c}e : {angle_deg:.2f}°\n"
            else : 
                non_contact_particules.append(p)
                if len(all_x_max) != 0 :
                    local_x_max = max(all_x_max)
                else :
                    local_x_max = ax.get_xlim()[1] * 1.05
                p.tracer_trajectoire(ax, E_val, 0, local_x_max * 1.2)

        if len(all_x_max) > 0:
            ax.plot([0, max(all_x_max) * 1.2], [0, 0], c='black', linewidth=5, label='Échantillon')
            if pot_val != 0 :
                zoom_target_x = (min(all_x_max) + max(all_x_max)) * 0.5
                zoom_factor_x = (min(all_x_max) + max(all_x_max)) * 0.5
                zoom_factor_y = hauteur_initiale * 1.1
                zoom_factor = 10 ** (10 * ((1 / zoom_val) - 1))
                ax.set_xlim(zoom_target_x - zoom_factor * zoom_factor_x * 1.05,
                            zoom_target_x + zoom_factor * zoom_factor_x * 0.3)
                ax.set_ylim(-zoom_factor * zoom_factor_y * 0.05, zoom_factor * zoom_factor_y)
                ax.text(0.8, 0.5, texte_angles, transform=ax.transAxes,
                    fontsize=10,bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.legend()
        fig.canvas.draw_idle()

    tracer(slider_E.val, slider_zoom.val)

    def update_zoom(val : float) -> None:
        """ Fonction appelée lors du changement du slider de zoom. """
        tracer(slider_E.val, val)

    def update_E(val : float) -> None:
        """ Fonction appelée lors du changement du slider de champ E. """
        tracer(val, slider_zoom.val)

    slider_zoom.on_changed(update_zoom)
    slider_E.on_changed(update_E)

    plt.show()


if __name__ == '__main__' :
    rapports_mq, vo = [(1, 1), (2, 1), (3, 1)], 1e6
    pot_min, pot_max = -5000, 5000
    h_initiale = 0.1


    tracer_ensemble_trajectoires_dynamique(rapports_mq, vo, potentiel_min=pot_min, potentiel_max=pot_max, hauteur_initiale=h_initiale)