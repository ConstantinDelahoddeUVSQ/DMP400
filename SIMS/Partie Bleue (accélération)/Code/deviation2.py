# Objectif 2

import matplotlib.pyplot as plt
# Retrait de Slider car les widgets sont dans l'interface principale
# from matplotlib.widgets import Slider
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
            Masse (en u) / Charge (en e, unités de charge élémentaire) de la particule.
        v_initiale : float
            Vitesse initiale de la particule (en m/s).
        angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians.
        hauteur_initiale : float
            Coordonnée en y du point de départ (en m).
        """
        # --- MODIFICATION OBLIGATOIRE ---
        # Le calcul du rapport masse/charge utilisait eV au lieu de e (charge élémentaire)
        # et ne gérait pas charge nulle ou négative correctement.
        mass_u = masse_charge[0]
        charge_e = masse_charge[1]
        if charge_e == 0:
            raise ValueError("La charge de la particule ne peut pas être nulle.")
        self.mq = (mass_u * constants.u) / (charge_e * constants.e) # kg/C
        # --- FIN MODIFICATION ---

        self.vo = v_initiale
        self.angle = angle_initial
        self.height = hauteur_initiale
        # Gardons m et c pour l'affichage potentiel dans les labels
        self.m = mass_u
        self.c = charge_e # Note : 'c' pour charge est ambigu, mais gardons-le pour la compatibilité avec le code existant

    def equation_trajectoire(self, x : float, E : float) -> float:
        """
        Equation de la trajectoire de la particule y(x)

        Parameters
        ----------
        x : float
            L'abscisse à laquelle on veut calculer la coordonnée en y
        E : float
            Valeur du champ électrique (en V/m), convention : positif si dirigé vers +y

        Returns
        -------
        float
            Coordonée y de la particule au point x
        """
        # Attention à la direction de E. Si E est V/d avec V potentiel de la plaque à y=0
        # et 0V à y=distance, E est dirigé vers +y.
        # L'accélération a_y = q*E/m. Si q est positive, a_y a le signe de E.
        # Formule standard y(t) = y0 + vy0*t + 0.5*ay*t^2
        # x(t) = x0 + vx0*t => t = (x-x0)/vx0
        # Ici x0=0. vx0 = v0*sin(angle vs y). vy0 = v0*cos(angle vs y)
        # y(x) = height + vy0 * (x/vx0) + 0.5 * (q*E/m) * (x/vx0)^2
        # y(x) = height + v0*cos(angle) * (x / (v0*sin(angle))) + 0.5 * (E/mq) * (x / (v0*sin(angle)))^2
        # y(x) = height + (x / tan(angle)) + 0.5 * (E/mq) * (x / (v0*sin(angle)))^2
        # Vérifions la formule du code :
        # X = x / (self.vo * np.sin(self.angle))
        # return 0.5 * E / self.mq * X * X - self.vo * np.cos(self.angle) * X + self.height
        # Il semble y avoir un signe moins sur le terme en cos(angle).
        # Si l'angle est défini par rapport à +y, et +y est vers le haut:
        # v_x = v0 * sin(angle)
        # v_y = v0 * cos(angle)
        # Si le champ E est dirigé vers +y (potentiel plus bas en haut), la force F_y = qE. a_y = qE/m = E/mq.
        # y(t) = height + v0*cos(angle)*t + 0.5*(E/mq)*t^2
        # x(t) = v0*sin(angle)*t => t = x / (v0*sin(angle))
        # y(x) = height + v0*cos(angle)*[x/(v0*sin(angle))] + 0.5*(E/mq)*[x/(v0*sin(angle))]^2
        # y(x) = height + x/tan(angle) + 0.5*(E/mq)*(x / (v0*sin(angle)))^2
        # La formule originale a un signe moins devant le terme cosinus, ce qui serait correct si
        # vy0 était dirigé vers le bas (-y), ou si l'angle était défini différemment.
        # Ou si le champ E était dirigé vers -y.
        # L'interface calcule E = Potentiel/distance. Si Potentiel est négatif (plaque à y=0), E est négatif (dirigé vers -y).
        # Si E est négatif, ay = q*E/m. Pour q>0, ay est négative.
        # y(t) = height + v0*cos(angle)*t + 0.5*(E/mq)*t^2 (E est négatif)
        # y(x) = height + x/tan(angle) + 0.5*(E/mq)*(x / (v0*sin(angle)))^2 (E est négatif)
        # La formule dans le code semble supposer E positif (magnitude) et ajoute le signe q/m manuellement.
        # Pour rester cohérent avec l'interface qui calcule E signé (V/d):
        vx0 = self.vo * np.sin(self.angle)
        vy0 = self.vo * np.cos(self.angle)
        if vx0 == 0: # Tir vertical
             # Gérer ce cas séparément si nécessaire, ici on retourne NaN pour éviter la division par zéro
             if isinstance(x, np.ndarray): return np.full(x.shape, np.nan)
             else: return np.nan
        t = x / vx0
        # a_y = qE/m = E / mq (car mq = m/q)
        y = self.height + vy0 * t + 0.5 * (E / self.mq) * t**2
        return y

    def trajectoire(self, E : float, x_min : float, x_max : float, n_points : int = 10000) -> tuple[np.ndarray, np.ndarray] :
        """
        Calcule la trajectoire entre un x minimum et un x maximum

        Parameters
        ----------
        E : float
            Valeur du champ électrique (en V/m), signé.
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
        y = self.equation_trajectoire(x, E)
        # Empêcher le tracé si la trajectoire est invalide (NaN)
        mask = ~np.isnan(y)
        return x[mask], y[mask]

    def tracer_trajectoire(self, ax, E : float, x_min : float, x_max : float, n_points : int = 10000) -> None :
        """
        Trace la trajectoire entre x_min et x_max sur ax

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            L'axe matplotlib sur lequel on veut tracer la trajectoire
        E : float
            Valeur du champ électrique (en V/m), signé.
        x_min : float
            Position en x minimale (en m)
        x_max : float
            Position en x maximale (en m)
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max
        """
        x, y = self.trajectoire(E, x_min, x_max, n_points)
        if len(x) > 0 : # Ne trace rien si la trajectoire est vide (e.g., tir vertical)
            ax.plot(x, y, label=f"Trajectoire {self.m}u, {self.c}e") # Mis à jour charge en 'e'


    def point_contact(self, E : float) -> float | None :
        """
        Calcule l'abscisse x où la particule atteint y=0 (touche la plaque).

        Parameters
        ----------
        E : float
            Valeur du champ électrique (en V/m), signé.

        Returns
        -------
        float or None
            Abscisse x du point de contact (si elle existe et est positive), sinon None.
        """
        # Résoudre y(x) = 0 pour x.
        # y(x) = height + x/tan(angle) + 0.5*(E/mq)*(x / (v0*sin(angle)))^2 = 0
        # Posons vx0 = v0*sin(angle), vy0 = v0*cos(angle), ay = E/mq
        # height + vy0*(x/vx0) + 0.5*ay*(x/vx0)^2 = 0
        # C'est une équation du second degré en x:
        # (0.5*ay/vx0^2) * x^2 + (vy0/vx0) * x + height = 0
        # a*x^2 + b*x + c = 0
        # avec a = 0.5*ay/vx0^2 = 0.5*(E/mq) / (v0*sin(angle))^2
        #      b = vy0/vx0 = 1/tan(angle)
        #      c = height

        vx0 = self.vo * np.sin(self.angle)
        vy0 = self.vo * np.cos(self.angle)

        if vx0 == 0: # Tir vertical, pas de contact en x>0
            return None

        a = 0.5 * (E / self.mq) / (vx0**2)
        # Gérer le cas d'un champ nul (trajectoire droite)
        if abs(a) < 1e-15: # Si a est très proche de zéro
            if abs(vy0) < 1e-15: # Tir horizontal (angle=pi/2)
                return None # Jamais de contact si height > 0
            else:
                # y = height + vy0*t = height + vy0*(x/vx0) = 0
                # x = -height * vx0 / vy0 = -height * tan(angle)
                x_contact = -self.height * (vx0 / vy0)
                return x_contact if x_contact > 0 else None
        else:
            # Equation quadratique standard
            b = vy0 / vx0
            c = self.height
            delta = b**2 - 4*a*c

            if delta < 0: # Pas de solution réelle
                return None
            else:
                # Deux solutions potentielles
                sqrt_delta = np.sqrt(delta)
                x1 = (-b + sqrt_delta) / (2*a)
                x2 = (-b - sqrt_delta) / (2*a)

                # On cherche la solution positive physiquement pertinente (généralement la plus petite positive)
                solutions_positives = []
                if x1 > 1e-9: solutions_positives.append(x1) # Utiliser une petite tolérance > 0
                if x2 > 1e-9: solutions_positives.append(x2)

                if not solutions_positives:
                    return None
                else:
                    # Retourner la plus petite solution positive
                    return min(solutions_positives)


    def angle_incident(self, E : float) -> float | None:
        """
        Calcule l'angle que la tangente à la trajectoire forme avec l'axe +x au point de contact (y=0).

        Parameters
        ----------
        E : float
            Valeur du champ électrique (en V/m), signé.

        Returns
        -------
        float or None
            Angle en radians (entre -pi/2 et pi/2) si le contact existe, sinon None.
        """
        x_contact = self.point_contact(E)
        if x_contact is None:
            return None

        # Calcul de la dérivée y'(x) = dy/dx au point de contact
        # y(x) = height + x/tan(angle) + 0.5*(E/mq)*(x / (v0*sin(angle)))^2
        # dy/dx = 1/tan(angle) + (E/mq) * x / (v0*sin(angle))^2
        vx0 = self.vo * np.sin(self.angle)
        vy0 = self.vo * np.cos(self.angle)

        if vx0 == 0: # Cas vertical
             return None

        # Gérer le cas E=0
        if abs(E) < 1e-15 :
             dydx = vy0 / vx0 # Pente constante
        else:
             dydx = (vy0 / vx0) + (E / self.mq) * x_contact / (vx0**2)

        # L'angle alpha par rapport à l'axe +x est donné par arctan(dy/dx)
        alpha = np.arctan(dydx)
        return alpha


# --- MODIFICATION OBLIGATOIRE ---
# Modification de la signature pour correspondre à l'appel depuis l'interface
# Suppression des paramètres surface et charge_plaque, ajout de E
# Ajout de l'argument ax
def tracer_ensemble_trajectoires(
        masse_charge_particules : list[tuple[int, int]],
        vitesse_initiale : float,
        angle_initial : float, # Reçoit l'angle en radians
        hauteur_initiale : float, # Reçoit la hauteur initiale
        E : float, # Reçoit le champ E calculé par l'interface
        ax # Reçoit l'axe Matplotlib de l'interface
        ) -> None :
    """
    Trace les trajectoires jusqu'au contact de différentes particules sur un axe donné.

    Parameters
    ----------
    masse_charge_particules : list of tupleof int
        Liste de tuples (Masse en u, Charge en e).
    vitesse_initiale : float
        Vitesse initiale commune à toutes les particules (m/s).
    angle_initial : float
        Angle initial commun par rapport à l'axe +y (radians).
    hauteur_initiale : float
        Coordonnée y initiale commune (m).
    E : float
        Valeur du champ électrique (V/m), signé (calculé par l'interface).
    ax : matplotlib.axes.Axes
        L'axe Matplotlib sur lequel tracer.
    """
    # Pas besoin de créer fig, ax ou sliders ici. Ils sont gérés par l'interface.

    # Utilise directement les paramètres reçus
    particules = [particule(mq, vitesse_initiale, angle_initial, hauteur_initiale) for mq in masse_charge_particules]

    all_x_max = []
    texte_angles = "Angles incidents (vs +x) :\n" # Préciser la référence de l'angle

    ax.axhline(0, color='black', linewidth=2, label='Échantillon (y=0)') # Trace l'axe y=0

    for p in particules:
        try:
            x_contact = p.point_contact(E)
            if x_contact is not None and x_contact > 0:
                all_x_max.append(x_contact)
                p.tracer_trajectoire(ax, E, 0, x_contact) # Trace jusqu'au contact

                angle_inc = p.angle_incident(E)
                if angle_inc is not None:
                    angle_deg = np.degrees(angle_inc)
                    texte_angles += f"- {p.m}u, {p.c}e : {angle_deg:.2f}°\n"
                else:
                     texte_angles += f"- {p.m}u, {p.c}e : Contact? Angle N/A\n"

            else: # Si pas de contact ou contact en x<=0, tracer sur une distance arbitraire?
                  # Pour l'instant, on ne trace rien si pas de contact x>0.
                  # Alternative : tracer sur une plage x fixe ?
                  # p.tracer_trajectoire(ax, E, 0, 0.1) # Exemple: tracer sur 10cm
                  texte_angles += f"- {p.m}u, {p.c}e : Pas de contact (x>0)\n"
                  pass # Ne rien tracer si pas de contact pertinent

        except ValueError as e:
             print(f"Erreur pour particule {p.m}u, {p.c}e: {e}")
             texte_angles += f"- {p.m}u, {p.c}e : Erreur ({e})\n"


    # Gérer les limites et l'affichage
    if len(all_x_max) > 0:
        max_x_contact = max(all_x_max)
        ax.set_xlim(0, max_x_contact * 1.1) # Ajuster xlim au contact le plus lointain
    else:
        ax.set_xlim(0, 0.1) # Limite par défaut si aucune particule ne touche

    # Ajuster ylim pour voir le départ et l'arrivée
    ax.set_ylim(min(0, hauteur_initiale * (-0.1)), hauteur_initiale * 1.1)

    # Afficher le texte des angles (positionnement à ajuster si besoin)
    ax.text(0.95, 0.95, texte_angles, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title("Déviation par Champ Électrique")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

    # PAS de plt.show() ni de fig.canvas.draw_idle() ici.
    # L'interface principale s'en charge.

# --- FIN MODIFICATION ---


if __name__ == '__main__':
    # Test simple qui ne crée plus de widgets internes
    rapports_mq = [(1, 1), (2, 1), (4, 2)] # Exemple: H+, D+, He++
    vo = 1e5 # m/s
    angle_rad = np.radians(30) # 30 degrés vs y
    y0 = 0.05 # 5 cm
    # E = -10000 / 0.05 # Champ pour -10kV sur la plaque à y=0, E dirigé vers -y
    potentiel_plaque = -5000 # V
    distance_plaques = 0.05 # m
    E_calc = champ_electrique_v2(distance_plaques, potentiel_plaque)

    # Pour tester, on crée une figure et un axe ici
    fig_test, ax_test = plt.subplots(figsize=(8, 6))

    print(f"Test avec V0={vo:.1e} m/s, Angle={np.degrees(angle_rad):.1f}°, y0={y0} m, E={E_calc:.1f} V/m")

    tracer_ensemble_trajectoires(rapports_mq, vo, angle_rad, y0, E_calc, ax=ax_test)

    # Afficher le graphique de test
    plt.show()