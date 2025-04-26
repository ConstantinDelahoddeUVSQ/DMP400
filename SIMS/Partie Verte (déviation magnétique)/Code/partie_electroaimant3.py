
# Objectif 1

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
            Masse (en unités atomiques), Charge (en unités de charge élémentaire, e) de la particule.
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s).
        """
        # --- MODIFICATION OBLIGATOIRE ---
        mass_u = masse_charge[0]
        charge_e = masse_charge[1]
        if charge_e == 0:
             raise ValueError("La charge de la particule ne peut pas être nulle.")
        # Utilisation de constants.e (charge élémentaire en C) au lieu de constants.eV
        self.mq = (mass_u * constants.u) / (charge_e * constants.e) # kg/C
        # --- FIN MODIFICATION ---

        self.vo = v_initiale
        self.m = mass_u
        self.c = charge_e # Garde la charge en unités 'e' pour l'affichage


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
        prefix = self.mq / Bz # C'est R = rayon de Larmor * signe(q*Bz)
        arg_arccos = 1 - x / (self.vo * prefix)

        # --- Gestion des erreurs arccos ---
        # Clipper l'argument pour éviter les erreurs dues aux imprécisions numériques
        arg_arccos = np.clip(arg_arccos, -1.0, 1.0)
        # --- Fin Gestion ---

        # L'angle phi de rotation est arccos(1 - x/R)
        # y = R * sin(phi)
        return self.vo * prefix * np.sin(np.arccos(arg_arccos))


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
        y = self.equation_trajectoire(x, Bz)
        # Filtrer les NaN qui pourraient apparaître si Bz=0 ou si x > diamètre
        mask = ~np.isnan(y)
        return x[mask], y[mask]


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
        if len(x) > 0:
            # --- MODIFICATION OBLIGATOIRE ---
            # Correction du label pour afficher 'e' au lieu de 'eV'
            ax.plot(x, y, label=f'{self.m}u, {self.c:+}e') # :+ pour afficher le signe de la charge
            # --- FIN MODIFICATION ---


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
        # Note: fsolve peut avoir des difficultés si l'équation a plusieurs solutions ou des discontinuités.
        # La valeur initiale B0 peut être importante.
        # Si mq est négatif, le sens de déviation change.
        if B0 is None :
            # Heuristique simple pour B0 : basé sur le rayon R approx y_objective (si x petit) ou x_objective/2 (si demi-cercle)
            # R = vo * mq / B => B = vo * mq / R
            # Utilisons une estimation basée sur R ~ sqrt(x^2+y^2) ? Non, plus complexe.
            # Tentons une valeur basée sur mq, mais en valeur absolue pour éviter B0 négatif
            B0 = abs(self.vo * self.mq / max(x_objective, y_objective, 1e-6)) # Estimation grossière
            if B0 == 0: B0 = 1e-3 # Éviter B0 nul

        # Définir l'équation à résoudre : y_calc(B) - y_objective = 0
        def equation_func(B):
            if abs(B) < 1e-15: return np.inf # Éviter division par zéro
            prefix = self.mq / B
            arg_arccos = 1 - x_objective / (self.vo * prefix)
            # Vérifier le domaine de arccos
            if not (-1.0 <= arg_arccos <= 1.0):
                # Retourner une grande valeur si hors domaine pour guider fsolve
                return 1e10 * np.sign(arg_arccos)
            y_calc = self.vo * prefix * np.sin(np.arccos(arg_arccos))
            return y_calc - y_objective

        # Utiliser fsolve pour trouver B
        B_solution, infodict, ier, mesg = fsolve(equation_func, B0, full_output=True)

        # Vérifier si fsolve a convergé
        if ier == 1:
            return B_solution[0]
        else:
            print(f"Avertissement: fsolve n'a pas convergé pour déterminer Bz ({mesg})")
            # Essayer une autre valeur initiale?
            # Tenter avec -B0 si la première tentative échoue?
            B0_alt = -B0
            B_solution_alt, infodict_alt, ier_alt, mesg_alt = fsolve(equation_func, B0_alt, full_output=True)
            if ier_alt == 1:
                 print(f"   -> Convergence réussie avec B0={B0_alt:.2e}")
                 return B_solution_alt[0]
            else:
                 print(f"   -> Échec aussi avec B0={B0_alt:.2e}")
                 return np.nan # Retourner NaN si aucune solution trouvée

# Niveau 2.2 : Tracer l'ensemble des trajectoires des particules d'un faisceau
def tracer_ensemble_trajectoires(masses_charges_particules : list[tuple[int, int]], vitesse_initiale : float, Bz : float, x_detecteur : float, create_plot : bool = True, ax = None) -> None:
    """
    Trace les trajectoires entre 0 et x_detecteur pour un ensemble de particules d'un faisceau

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
    if ax == None or create_plot == True :
        fig, ax = plt.subplots()
    
    all_y_contact = []
    for particule_locale in particules :
        particule_locale.tracer_trajectoire(ax, Bz, 0, x_detecteur)
        all_y_contact.append(particule_locale.equation_trajectoire(x_detecteur, Bz))

    ax.plot([x_detecteur, x_detecteur], [min(all_y_contact) * 0.8, max(all_y_contact) * 1.1], c='black', linewidth=5, label='Détecteur')
    ax.set_xlabel('Position x (en m)')
    ax.set_ylabel('Position y (en m)')
    ax.legend()
    if create_plot :
        plt.show()


def tracer_trajectoires_dynamiquement(masses_charges_particules : list[tuple[int, int]], vi_min : float, vi_max : float, Bz_min : float, Bz_max : float, x_detecteur : float, create_plot : bool = True, fig = None, ax = None) -> None:
    """
    Trace les trajectoires entre 0 et x_detecteur pour un ensemble de particules d'un faisceau de manière dynamique

    Parameters
    ----------
    masses_charges_particules : list of tuple of int
        Masse (en unités atomiques), Charge (en eV)  pour toutes les particules
    vi_min : float
        Vitesse intiale en y minimale (en m/s)
    vi_max : float
        Vitesse intiale en y maximale (en m/s)
    Bz_min : float
        Valeur minimale du champ magnétique d'axe z (en T)
    Bz_max : float
        Valeur maximale du champ magnétique d'axe z (en T)
    x_detecteur : float
        L'abscisse du détecteur (en m)
    """
    particules = [particule(masse_charge, 0.5 * (vi_min + vi_max)) for masse_charge in masses_charges_particules]
    if create_plot or ax == None : 
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)
    
    all_y_contact = []
    all_lines = []
    Bz0 = 0.5 * (Bz_min + Bz_max)
    for particule_locale in particules :
        all_lines.append(particule_locale.tracer_trajectoire(ax, Bz0, 0, x_detecteur))
        all_y_contact.append(particule_locale.equation_trajectoire(x_detecteur, Bz0))

    detecteur = ax.plot([x_detecteur, x_detecteur], [min(all_y_contact) * 0.8, max(all_y_contact) * 1.1], c='black', linewidth=5, label='Détecteur')

    ax.legend()

    ax_a = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_b = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_a = Slider(ax_a, 'V0 (m/s)', vi_min, vi_max, valinit= 0.5 * (vi_min + vi_max))
    slider_b = Slider(ax_b, 'Bz (T)', Bz_min, Bz_max, valinit= Bz0)

    def update(val) :
        v0 = slider_a.val
        Bz = slider_b.val
        particules = [particule(masse_charge, v0) for masse_charge in masses_charges_particules]
        for i in range(len(all_lines)) :
            all_lines[i][0].set_ydata(particules[i].trajectoire(Bz, 0, x_detecteur)[1])
        fig.canvas.draw_idle()
    
    slider_a.on_changed(update)
    slider_b.on_changed(update)
    return ax
    plt.show()



'''
Test de la fonction tracer_ensemble_trajectoires (valeurs non représentatives)

On trace les trajectoires de particules avec des rapports m/q différents dans un champ magnétique donné
'''
if __name__ == '__main__' :
    rapports_masse_charge = [(1, 1), (2, 1), (3, 1)]
    vitesse_initiale = 1e7
    Bz = 1
    x_detecteur = 1e-4
    
    tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_detecteur)


'''
Test de la fonction déterminer_champ_magnétique (valeurs non représentatives)

On cherche le champ magnétique pour dévier la trajectoire en x_max, x_max
Puis on trace la trajectoire jusqu'en x_max
On remarque que la particule finit effectivement à la position prévue
'''
# if __name__ == '__main__' :
#     rapports_masse_charge, vi = [(1, 1)], 1e7
#     p = particule(rapports_masse_charge[0], vi)
#     x_max = 4.95e-2
#     Bz = p.determiner_champ_magnetique(x_max, x_max)
#     print(Bz)
#     tracer_ensemble_trajectoires(rapports_masse_charge, vi, Bz, x_max)
    

"""
Test de la fonction tracer_trajectoires_dynamiquement (valeurs non représentatives)
"""
# if __name__ == '__main__' :
#     rapports_masse_charge = [(1, 1), (2, 1), (3, 1)]
#     vi_min, vi_max = 1e7, 1e8
#     Bz_min, Bz_max = 1, 5
#     x_detecteur = 4.95e-2
    
#     tracer_trajectoires_dynamiquement(rapports_masse_charge, vi_min, vi_max, Bz_min, Bz_max, x_detecteur)