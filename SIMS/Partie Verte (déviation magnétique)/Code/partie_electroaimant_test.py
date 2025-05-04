# Objectif 1

import matplotlib.pyplot as plt
# Retirer Slider si non utilisé
# from matplotlib.widgets import Slider
import numpy as np
from scipy.optimize import fsolve
import scipy.constants as constants
import itertools # Si besoin pour couleurs par défaut

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
        if charge_e == 0: raise ValueError("La charge ne peut pas être nulle.")
        if mass_u <= 0: raise ValueError("La masse doit être positive.")
        if v_initiale <= 0: raise ValueError("La vitesse initiale doit être positive.")

        # mq est utilisé pour le rayon R = mq * v / B, donc doit utiliser |q|
        self.mq = (mass_u * constants.u) / (abs(charge_e) * constants.e) # kg/C (toujours positif)
        self.vo = v_initiale
        self.m = mass_u
        # Stocker la charge signée pour l'affichage
        self.charge_affichage = charge_e

    def equation_trajectoire(self, x : float | np.ndarray, Bz : float) -> float | np.ndarray:
        """
        Calcule la position y pour une position x donnée.
        Formule équivalente à y = sqrt(2Rx - x^2) où R = mq*vo/Bz.
        Attention: Bz ne doit pas être nul.
        L'équation suppose une déviation vers x positif. Le signe de q*Bz détermine
        si la déviation est réellement vers +x ou -x, mais la formule donne la forme.

        Parameters
        ----------
        x : float or np.ndarray
            Position(s) en x (en m). Doit être dans [0, 2R].
        Bz : float
            Valeur du champ magnétique (en T). Doit être != 0.

        Returns
        -------
        float or np.ndarray
            Position(s) en y (en m). NaN si x est hors domaine.
        """
        if abs(Bz) < 1e-15:
             # Trajectoire droite si Bz=0
             if isinstance(x, np.ndarray): return np.zeros_like(x) # Ou plutôt y=vo*t, x=0? Non, v0 est en y.
             else: return 0.0 # y reste 0 si v0 en y et B=0? Non plus.
             # Si v0 est en y, et B=0, la particule continue en ligne droite sur l'axe y.
             # La fonction y(x) n'est pas définie sauf pour x=0 où y=0.
             # Retourner NaN est peut-être le plus sûr pour indiquer un problème.
             if isinstance(x, np.ndarray): return np.full_like(x, np.nan)
             else: return np.nan


        # R = self.mq * self.vo / abs(Bz) # Le rayon dépend de |B|
        # Pour être cohérent avec la formule utilisant arccos(1-x/prefix)
        # où prefix = mq/Bz, il faut que le signe de Bz soit pris en compte dans prefix.
        # Si Bz > 0 et q > 0, force vers +x. Si Bz < 0 et q > 0, force vers -x.
        # Si q < 0, c'est inversé. La formule donne y>=0, donc elle représente
        # la forme de la déviation, pas nécessairement la direction exacte sans
        # considérer q*Bz. Pour l'instant, on garde la formule telle quelle.
        prefix = self.mq / Bz # Attention, mq>0. Signe de prefix = signe de Bz.

        # Le rayon de courbure physique est R = mq*vo/|Bz|
        # L'argument de arccos est 1 - x / prefix = 1 - x*Bz / mq
        # Il doit être entre -1 et 1.
        # 1 - x*Bz/mq <= 1  => -x*Bz/mq <= 0 => x*Bz >= 0.
        # 1 - x*Bz/mq >= -1 => 2 >= x*Bz/mq => 2*mq/Bz >= x (si Bz>0) ou 2*mq/Bz <= x (si Bz<0) ?
        # C'est plus simple avec R: l'argument est 1 - x*sgn(Bz)/R
        # 1 - x*sgn(Bz)/R <= 1 => -x*sgn(Bz)/R <= 0 => x*sgn(Bz) >= 0
        # 1 - x*sgn(Bz)/R >= -1 => 2 >= x*sgn(Bz)/R => 2R >= x*sgn(Bz) => x <= 2R si sgn(Bz)>0, x >= -2R si sgn(Bz)<0?
        # La formule y=sqrt(2Rx-x^2) est pour x dans [0, 2R]. Utilisons cela.
        R = self.mq * self.vo / abs(Bz)
        # Calculer y seulement pour x dans le domaine valide
        x_valid = np.where((x >= 0) & (x <= 2 * R), x, np.nan)

        with np.errstate(invalid='ignore'): # Ignore sqrt(negatif) qui donnera NaN
            y = np.sqrt(2 * R * x_valid - x_valid**2)
        return y

    def trajectoire(self, Bz : float, x_min : float, x_max : float, n_points : int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Calcule les points (x, y) de la trajectoire."""
        if x_max <= x_min: return np.array([]), np.array([])
        x = np.linspace(x_min, x_max, n_points)
        y = self.equation_trajectoire(x, Bz)
        # Retourner seulement les points valides (où y n'est pas NaN)
        mask = ~np.isnan(y)
        return x[mask], y[mask]

    def tracer_trajectoire(self, ax, Bz : float, x_min : float, x_max : float, color=None, label=None, n_points : int = 1000) -> None:
        """Trace la trajectoire sur ax en utilisant couleur et label."""
        x, y = self.trajectoire(Bz, x_min, x_max, n_points)
        if len(x) == 0: return # Ne rien tracer si vide

        plot_kwargs = {'c': color if color else None}
        if label: plot_kwargs['label'] = label

        ax.plot(x, y, **plot_kwargs)

    def determiner_champ_magnetique(self, x_objective : float, y_objective : float, B0 : float = None) -> float | None:
        """Donne Bz pour atteindre (x, y). Retourne None si impossible."""
        if x_objective <= 0 or y_objective < 0:
             print("Avertissement: Objectif (x,y) doit être dans le quadrant x>0, y>=0.")
             return None
        if y_objective == 0: # Arrivée sur l'axe x
             # y = sqrt(2Rx - x^2) = 0 => 2Rx - x^2 = 0 => x(2R-x)=0
             # Donc soit x=0 (départ), soit x=2R.
             # On cherche B tel que 2R = x_objective
             # 2 * (mq*vo / |B|) = x_objective => |B| = 2*mq*vo / x_objective
             # Il faut choisir le signe. Si q>0, déviation vers +x => Bz>0. Si q<0, déviation vers -x? Non, v0 // y.
             # Si v0 // +y et B // +z, force F ~ q (v ^ B) ~ q (uy ^ uz) ~ q (-ux). Force vers -x si q>0.
             # Si v0 // +y et B // -z, force F ~ q (uy ^ -uz) ~ q (+ux). Force vers +x si q>0.
             # Donc pour atteindre x_objective > 0 avec q>0, il faut Bz < 0.
             # Et pour atteindre x_objective > 0 avec q<0, il faut Bz > 0.
             signe_q = np.sign(self.charge_affichage)
             signe_B = -signe_q
             return signe_B * (2 * self.mq * self.vo / x_objective)

        # Cas général y_objective > 0
        # y^2 = 2Rx - x^2 => y^2 = 2*(mq*vo/|B|)*x - x^2
        # y^2 + x^2 = 2*mq*vo*x / |B|
        # |B| = 2*mq*vo*x / (x^2 + y^2)
        # Choisir le signe comme avant
        signe_q = np.sign(self.charge_affichage)
        signe_B = -signe_q
        # Utiliser les objectifs x, y
        abs_B = (2 * self.mq * self.vo * x_objective) / (x_objective**2 + y_objective**2)
        return signe_B * abs_B
        # La méthode fsolve est plus générale mais peut échouer ou converger vers mauvaise solution.
        # L'expression analytique est préférable ici.

# --- Fonction de Traçage Ensemble (Adaptée) ---

def tracer_ensemble_trajectoires(
        masses_charges_particules : list[tuple[float, float]], # Accepte (m, c)
        vitesse_initiale : float,
        Bz : float,
        x_detecteur : float,
        labels_particules: list[str] = None, # NOUVEAU: Noms des particules
        create_plot : bool = True,
        ax = None
    ) -> None:
    """
    Trace les trajectoires pour un ensemble de particules avec labels fournis.
    """
    if ax is None or create_plot :
        fig, ax = plt.subplots(figsize=(10, 8))
    if labels_particules is None: labels_particules = [f"Particule {i+1}" for i in range(len(masses_charges_particules))]
    if len(labels_particules) != len(masses_charges_particules):
        print("Avertissement: Noms/Particules mismatch.")
        labels_particules = [f"{mc[0]:.1f}u,{mc[1]:+.0f}e" for mc in masses_charges_particules]

    particules_objs = [particule(mc, vitesse_initiale) for mc in masses_charges_particules]
    all_y_contact = []
    plotted_lines = [] # Pour obtenir les couleurs si besoin

    for i, p_local in enumerate(particules_objs):
        label = labels_particules[i] # Utiliser le nom fourni
        try:
            y_contact = p_local.equation_trajectoire(x_detecteur, Bz)
            p_local.tracer_trajectoire(ax, Bz, 0, x_detecteur, label=label)
            # Récupérer la ligne tracée pour info couleur (si besoin plus tard)
            # plotted_lines.append(ax.get_lines()[-1])
            if y_contact is not None and not np.isnan(y_contact):
                all_y_contact.append(y_contact)
            else: # Si NaN (hors domaine ou Bz=0)
                 all_y_contact.append(np.nan) # Garder trace qu'on n'a pas de y valide

        except ValueError as e:
             print(f"Erreur calcul/tracé pour {label}: {e}")
             all_y_contact.append(np.nan)

    # Tracer le détecteur en ajustant sa hauteur
    valid_y_contact = [y for y in all_y_contact if y is not None and not np.isnan(y)]
    if valid_y_contact:
        min_y = min(valid_y_contact)
        max_y = max(valid_y_contact)
        marge_y = abs(max_y - min_y) * 0.1 + 0.01 # Marge + petit offset
        y_det_min = min_y - marge_y
        y_det_max = max_y + marge_y
        ax.plot([x_detecteur, x_detecteur], [y_det_min, y_det_max], c='black', linewidth=3, label='Détecteur')
        # Ajuster les limites y
        ymin_plot, ymax_plot = ax.get_ylim()
        ax.set_ylim(min(ymin_plot, y_det_min), max(ymax_plot, y_det_max))
    else: # Si aucune particule n'atteint x_detecteur
         # Tracer un détecteur par défaut près de y=0 ?
         ax.plot([x_detecteur, x_detecteur], [-0.01, 0.01], c='black', linewidth=3, label='Détecteur')

    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Position y (m)')
    ax.set_title(f"Déviation Magnétique (Bz={Bz:.3f} T)")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

    if create_plot : plt.show()

def tracer_trajectoires_dynamiquement(masses_charges_particules : list[tuple], vi_min : float, vi_max : float, Bz_min : float, Bz_max : float, x_detecteur : float) -> None:
    """
    Trace les trajectoires entre 0 et x_detecteur pour un ensemble de particules d'un faisceau de manière dynamique

    Parameters
    ----------
    masses_charges_particules : list of tuple
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
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    all_y_contact = []
    all_lines = []
    Bz0 = 0.5 * (Bz_min + Bz_max)
    for particule_locale in particules :
        y_contact = particule_locale.equation_trajectoire(x_detecteur, Bz0)
        all_y_contact.append(y_contact)
        label = ''
        if np.isnan(y_contact) :
            label = ' ; Pas de contact'
        all_lines.append(particule_locale.tracer_trajectoire(ax, Bz0, 0, x_detecteur, add_label=label))
        

    detecteur = ax.plot([x_detecteur, x_detecteur], [ax.get_ybound()[0], ax.get_ybound()[1]], c='black', linewidth=5, label='Détecteur')

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

    plt.show()


# --- Bloc Test ---
if __name__ == '__main__':
    mq_list = [(1, 1), (2, 1), (4, -2)] # H+, D+, He-- ? Test avec charge neg
    vo = 1e6
    bz = -0.2 # Bz négatif pour dévier H+ et D+ vers +x
    x_det = 0.1
    noms = ["Proton (H+)", "Deutéron (D+)", "Alpha--?"] # Noms exemples

    fig_test, ax_test = plt.subplots()
    tracer_ensemble_trajectoires(mq_list, vo, bz, x_det, labels_particules=noms, ax=ax_test, create_plot=False)
    ax_test.set_title("Test partie_electroaimant avec noms")
    plt.show()

'''
Test de la fonction tracer_ensemble_trajectoires (valeurs non représentatives)
On trace les trajectoires de particules avec des rapports m/q différents dans un champ magnétique donné
'''
# if __name__ == '__main__' :
#     rapports_masse_charge = [(1, 1), (2, 1), (3, 1)]
#     vitesse_initiale = 1e7
#     Bz = 1
#     x_detecteur = 1e-4
    
#     tracer_ensemble_trajectoires(rapports_masse_charge, vitesse_initiale, Bz, x_detecteur)


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
#     vi_min, vi_max = 1e4, 1e6
#     Bz_min, Bz_max = 1, 5
#     x_detecteur = 0.1
    
#     tracer_trajectoires_dynamiquement(rapports_masse_charge, vi_min, vi_max, Bz_min, Bz_max, x_detecteur)