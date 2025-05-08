# --- START OF FILE deviation_final.py ---

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import itertools # Pour les couleurs dans les fonctions potentiel

# Optionnel: import incertitude # Décommenter si besoin futur
# from matplotlib.widgets import Slider # Retiré car géré par l'UI principale

# --- Fonctions Utilitaires ---

def champ_electrique_v2(distance: float, difference_potentiel: float) -> float:
    """
    Calcule le champ électrique uniforme entre deux plaques parallèles.
    E = V/d, où V est le potentiel de la plaque à y=0 par rapport à y=distance.
    Un potentiel négatif donne un E négatif (dirigé vers -y).

    Parameters
    ----------
    distance : float
        Distance entre les plaques (en m). > 0.
    difference_potentiel : float
        Différence de potentiel V(y=0) - V(y=distance) (en V).

    Returns
    -------
    float
        Intensité du champ électrique E (en V/m). Signé.

    Raises
    ------
    ValueError
        Si la distance est nulle ou négative.
    """
    if distance <= 0:
        raise ValueError("La distance doit être strictement positive.")
    return difference_potentiel / distance

# --- Classe Particule ---

class particule:
    def __init__(self, masse_charge : tuple[float, float], v_initiale : float = 0, angle_initial : float = np.pi / 4, hauteur_initiale : float = 0.5, is_incertitude : bool = False, incertitude_unique : bool = False, base_mq : tuple = None) -> None :
        # ... (code __init__ précédent, s'assurer que self.angle = angle_initial) ...
        mass_u, charge_e = masse_charge
        if charge_e == 0: raise ValueError("La charge ne peut pas être nulle.")
        if mass_u <= 0: raise ValueError("La masse doit être positive.")
        if v_initiale < 0: raise ValueError("La vitesse initiale ne peut être négative.")
        # Validation angle critique pour incertitudes
        # On la garde stricte ici, mais create_incertitude_params bornera
        if not (0 < angle_initial < np.pi/2): raise ValueError("L'angle initial doit être entre 0 et pi/2 radians (exclus).")
        if hauteur_initiale <= 0: raise ValueError("La hauteur initiale doit être positive.")

        self.mq = (mass_u * constants.u) / (charge_e * constants.e) # kg/C (signé!)
        self.vo = v_initiale
        self.angle = angle_initial # Angle vs +y <<< Garder cette ligne
        self.height = hauteur_initiale
        self.m = mass_u
        self.c = charge_e
        self.is_incertitude = is_incertitude
        self.incertitude_unique = incertitude_unique
        self.base_mq = base_mq if base_mq else (mass_u, charge_e)


    def equation_trajectoire(self, x : float | np.ndarray, E : float) -> float | np.ndarray:
        """
        Equation de la trajectoire y(x) pour un champ E constant selon +y.
        Basée sur y(t) = h0 + vy0*t + 0.5*ay*t^2, avec vy0 = -vo*cos(angle vs +y).
        """
        angle = self.angle # Angle vs +y, dans ]0, pi/2[
        h0 = self.height
        v0 = self.vo
        mq = self.mq

        vx0 = v0 * np.sin(angle)
        vy0 = -v0 * np.cos(angle) # Vers le bas
        ay = E / mq

        if abs(vx0) < 1e-15:
            return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan

        t = x / vx0
        y = h0 + vy0 * t + 0.5 * ay * t**2
        return y

    def trajectoire(self, E : float, x_min : float, x_max : float, n_points : int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Calcule les points (x, y) de la trajectoire."""
        # ... (code inchangé) ...
        if x_max <= x_min: return np.array([]), np.array([])
        x = np.linspace(x_min, x_max, n_points)
        y = self.equation_trajectoire(x, E)
        mask = ~np.isnan(y)
        return x[mask], y[mask]


    def point_contact(self, E : float) -> float | None:
        """
        Calcule l'abscisse x > 0 où la particule atteint y=0.
        Retourne None si pas de contact ou contact en x <= 0.
        Utilise np.isfinite et tolérance delta.
        """
        vx0 = self.vo * np.sin(self.angle)
        # Correction: vy0 doit être négatif pour correspondre à la physique de l'équation
        # qui a été dérivée pour un départ vers le bas.
        vy0 = -self.vo * np.cos(self.angle) # <= Assurer la cohérence avec equation_trajectoire
        ay = E / self.mq

        # Vérifier validité des paramètres initiaux
        if not np.isfinite(vx0) or not np.isfinite(vy0) or not np.isfinite(ay):
            # print(f"Debug: Paramètres invalides pour point_contact - vx0={vx0}, vy0={vy0}, ay={ay}")
            return None
        if abs(vx0) < 1e-15: return None # Mouvement vertical

        a = 0.5 * ay / (vx0**2)
        b = vy0 / vx0 # = -1 / tan(angle)
        c = self.height

        # Vérifier si a, b, c sont finis
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
             # print(f"Debug: Coeffs invalides pour point_contact - a={a}, b={b}, c={c}")
             return None

        with np.errstate(divide='ignore', invalid='ignore'):
            if abs(a) < 1e-15: # Cas quasi-linéaire
                x_contact = -c / b if abs(b) > 1e-9 else None
            else:
                delta = b**2 - 4*a*c
                # --- MODIFICATION : Tolérance pour delta ---
                if delta < -1e-12: # Si clairement négatif (avec marge pour flottants)
                    x_contact = None
                else:
                    # Traiter delta proche de 0 comme 0 pour éviter erreurs sqrt
                    sqrt_delta = np.sqrt(max(0.0, delta))
                    # Calculer les racines
                    # Gérer division par zéro si 'a' est minuscule mais non nul
                    denom = 2*a
                    if abs(denom) < 1e-15:
                         x1 = np.inf if np.sign(-b + sqrt_delta) == np.sign(denom) else -np.inf
                         x2 = np.inf if np.sign(-b - sqrt_delta) == np.sign(denom) else -np.inf
                    else:
                         x1 = (-b + sqrt_delta) / denom
                         x2 = (-b - sqrt_delta) / denom
                    # --- FIN MODIFICATION ---

                    solutions_positives = []
                    tolerance = 1e-9
                    # --- MODIFICATION : Vérifier le type et la finitude AVANT la comparaison ---
                    if isinstance(x1, (int, float)) and np.isfinite(x1):
                        if x1 > tolerance:
                            solutions_positives.append(x1)
                    # else: # Debug si x1 n'est pas un scalaire fini
                    #     print(f"Debug point_contact: x1 non valide - type={type(x1)}, value={x1}")

                    if isinstance(x2, (int, float)) and np.isfinite(x2):
                        if x2 > tolerance:
                            solutions_positives.append(x2)
                    # else: # Debug
                    #     print(f"Debug point_contact: x2 non valide - type={type(x2)}, value={x2}")
                    # --- FIN MODIFICATION ---

                    x_contact = min(solutions_positives) if solutions_positives else None
        return x_contact

    def angle_incident(self, E : float) -> float | None:
        """Calcule l'angle (radians) de la tangente vs +x au point de contact."""
        # ... (code inchangé - utilise point_contact corrigé) ...
        x_contact = self.point_contact(E)
        if x_contact is None: return None

        vx0 = self.vo * np.sin(self.angle)
        vy0 = -self.vo * np.cos(self.angle)
        ay = E / self.mq

        if abs(vx0) < 1e-15: return None # Vertical

        dydx = (vy0 / vx0) + (ay * x_contact) / (vx0**2)
        alpha = np.arctan(dydx)
        return alpha


    def tracer_trajectoire(self, ax, E : float, x_min : float, x_max : float, color=None, label=None, is_uncertainty_plot=False, n_points : int = 1000) -> None:
        """Trace la trajectoire sur ax. Gère style/couleur."""
        # ... (code inchangé) ...
        x, y = self.trajectoire(E, x_min, x_max, n_points)
        if len(x) == 0: return
        plot_kwargs = {}
        plot_kwargs['c'] = color
        plot_kwargs['label'] = label
        if is_uncertainty_plot:
            plot_kwargs['linestyle'] = '--'; plot_kwargs['alpha'] = 0.7

        ax.plot(x, y, **plot_kwargs)


# ... (Fonction calculer_trajectoire_et_impact inchangée - utilise les méthodes corrigées) ...
def calculer_trajectoire_et_impact(
    # ... (code inchangé) ...
    masse_charge: tuple[float, float],
    vitesse_initiale: float,
    potentiel: float,
    angle_initial_rad: float,
    hauteur_initiale: float,
    n_points: int = 1000
) -> tuple[np.ndarray, np.ndarray, float | None]:
    try:
        p = particule(masse_charge, vitesse_initiale, angle_initial_rad, hauteur_initiale)
        E = champ_electrique_v2(hauteur_initiale, potentiel)
        x_impact = p.point_contact(E) # Utilise la méthode corrigée

        if x_impact is not None and x_impact > 0:
            x_traj, y_traj = p.trajectoire(E, 0, x_impact, n_points) # Utilise la méthode corrigée
        else:
            x_max_plot = max(hauteur_initiale * np.tan(angle_initial_rad) * 2, 0.1) if np.tan(angle_initial_rad)!=0 else 0.1
            x_traj, y_traj = p.trajectoire(E, 0, x_max_plot, n_points)
            x_impact = None

        return x_traj, y_traj, x_impact
    except ValueError as e:
        print(f"Erreur calcul traj.: {e}")
        return np.array([]), np.array([]), None

# ... (Fonction tracer_ensemble_trajectoires inchangée - utilise les méthodes corrigées) ...
def tracer_ensemble_trajectoires(
    # ... (code inchangé) ...
        masse_charge_particules : list[tuple[float, float]],
        vitesse_initiale : float,
        potentiel : float,
        angle_initial : float, # Radians
        hauteur_initiale : float,
        labels_particules: list[str] = None, # Liste des noms
        create_plot=True,
        ax=None
    ) -> None :
    if create_plot or ax is None : fig, ax = plt.subplots(figsize=(10, 8))
    if labels_particules is None: labels_particules = [f"Particule {i+1}" for i in range(len(masse_charge_particules))]
    if len(labels_particules) != len(masse_charge_particules):
        print("Avertissement: Noms/Particules mismatch.")
        labels_particules = [f"{mc[0]:.1f}u,{mc[1]:+.0f}e" for mc in masse_charge_particules] # Fallback labels

    E = champ_electrique_v2(hauteur_initiale, potentiel)
    all_x_max = []
    texte_angles = "Angles incidents (vs +x):"
    is_contact = False
    non_contact_list_info = [] # Stocker (particule, label)

    for i, mc in enumerate(masse_charge_particules):
        try:
            p = particule(mc, vitesse_initiale, angle_initial, hauteur_initiale)
            label = labels_particules[i]
            x_contact = p.point_contact(E)

            if x_contact is not None and x_contact > 0:
                all_x_max.append(x_contact)
                p.tracer_trajectoire(ax, E, 0, x_contact, label=label) # Utilise label fourni
                angle_inc = p.angle_incident(E) # Angle vs +x
                angle_deg = np.degrees(angle_inc) if angle_inc is not None else None
                texte_angles += f"\n- {label}: {angle_deg:.1f}°" if angle_deg is not None else f"\n- {label}: Contact?" # Garder tel quel
                is_contact = True
            else:
                texte_angles += f"\n- {label}: Pas de contact (x>0)"
                non_contact_list_info.append({'p': p, 'label': label}) # Garder pour tracer après xlim

        except ValueError as e:
            print(f"Erreur pour particule {mc}: {e}")
            texte_angles += f"\n- {labels_particules[i]}: Erreur"

    # Définir xlim avant de tracer les non-contacts
    if all_x_max: xlim_max = max(all_x_max) * 1.1
    else: xlim_max = hauteur_initiale * 2 # Limite par défaut si aucun contact
    ax.set_xlim(0, xlim_max)

    # Tracer les non-contacts
    for item in non_contact_list_info:
        p = item['p']; label = item['label']
        p.tracer_trajectoire(ax, E, 0, xlim_max, label=f"{label} (pas contact)")
        if xlim_max not in all_x_max: all_x_max.append(xlim_max)


    # Finalisation plot
    ax.plot([0, xlim_max], [0, 0], c='black', linewidth=3, label='Échantillon (y=0)')
    ax.text(0.98, 0.98, texte_angles, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Déviation Électrique (V={potentiel:.1f} V)")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()

# --- Fonctions Incertitudes (avec clamp_angle corrigé) ---
def create_incertitude_params(p : particule, incertitudes : dict, E : float) -> tuple:
    """Crée les particules bornes et E bornes."""
    def clamp_angle(angle, min_rad=np.radians(0.1), max_rad=np.radians(89.9)):
        return np.clip(angle, min_rad, max_rad)

    # Vérifier que les clés existent dans incertitudes ou utiliser 0
    m_inc = incertitudes.get('m', 0.0)
    q_inc = incertitudes.get('q', 0.0)
    v0_inc = incertitudes.get('v0', 0.0)
    theta_inc = incertitudes.get('theta', 0.0)
    h_inc = incertitudes.get('h', 0.0)
    E_inc = incertitudes.get('E', 0.0)

    # Calcul bornes params
    m_p, m_m = p.m * (1 + m_inc), p.m * (1 - m_inc)
    c_p, c_m = p.c * (1 + q_inc), p.c * (1 - q_inc) # Suppose q_inc positif
    v0_p, v0_m = p.vo * (1 + v0_inc), p.vo * (1 - v0_inc)
    a_p_raw, a_m_raw = p.angle * (1 + theta_inc), p.angle * (1 - theta_inc)
    a_p, a_m = clamp_angle(a_p_raw), clamp_angle(a_m_raw) # Borner
    h_p, h_m = p.height * (1 + h_inc), p.height * (1 - h_inc)
    E_p, E_m = E * (1 + E_inc), E * (1 - E_inc)

    # Validation
    if m_m <= 0 or v0_m < 0 or h_m <= 0:
        print("Avertissement: Incertitudes trop grandes, params invalides.")
        # Fallback: utiliser valeurs nominales pour les particules d'incertitude
        p_a = particule((p.m, p.c), p.vo, p.angle, p.height, True, True, (p.m, p.c))
        p_b = particule((p.m, p.c), p.vo, p.angle, p.height, True, False, (p.m, p.c))
        return p_a, p_b, E, E

    # Créer particules bornes
    p_a = particule((m_m, c_m), v0_m, a_m, h_m, True, True, (p.m, p.c))
    p_b = particule((m_p, c_p), v0_p, a_p, h_p, True, False, (p.m, p.c))

    return p_a, p_b, E_m, E_p # Retourne bornes E aussi

def tracer_ensemble_trajectoires_avec_incertitudes(
        # ... (Arguments inchangés) ...
        masse_charge_particules : list[tuple[float, float]],
        vitesse_initiale : float,
        incertitudes : dict,
        potentiel : float,
        angle_initial : float,
        hauteur_initiale : float,
        labels_particules: list[str] = None, # Liste des noms
        create_plot=True,
        ax=None
    ) -> None:
    # ... (Setup inchangé: créer fig/ax, vérifier labels, calculer E_nominal) ...
    if create_plot or ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    if labels_particules is None: labels_particules = [f"P{i+1}" for i in range(len(masse_charge_particules))]
    if len(labels_particules) != len(masse_charge_particules): labels_particules = [f"{mc[0]:.1f}u,{mc[1]:+.0f}e" for mc in masse_charge_particules]

    E_nominal = champ_electrique_v2(hauteur_initiale, potentiel)
    particules_base = [particule(mc, vitesse_initiale, angle_initial, hauteur_initiale) for mc in masse_charge_particules]

    all_x_max_global = []
    texte_angles = "Angles incidents (Nominal) (vs +x):"
    plotted_incert_labels = set()
    colors = plt.cm.viridis(np.linspace(0, 1, len(particules_base)))
    non_contact_nominal_info = [] # (p_base, color, label_base)
    non_contact_incert_info = [] # (p_inc, E_bound, color, label_to_use)

    for i, p_base in enumerate(particules_base):
        color = colors[i]; label_base = labels_particules[i]
        label_incert = f"Incert. {label_base}"

        # Tracer nominal
        x_contact_nom = p_base.point_contact(E_nominal)
        if x_contact_nom is not None and x_contact_nom > 0:
            all_x_max_global.append(x_contact_nom)
            p_base.tracer_trajectoire(ax, E_nominal, 0, x_contact_nom, color=color, label=label_base)
            angle_inc = p_base.angle_incident(E_nominal)
            angle_deg = np.degrees(angle_inc) if angle_inc is not None else None
            texte_angles += f"\n- {label_base}: {angle_deg:.1f}°" if angle_deg is not None else f"\n- {label_base}: Contact?"
        else:
            texte_angles += f"\n- {label_base}: Pas contact (x>0)"
            non_contact_nominal_info.append({'p':p_base, 'color':color, 'label':label_base})

        # Créer et tracer incertitudes
        try:
            p_inc_a, p_inc_b, E_bound_a, E_bound_b = create_incertitude_params(p_base, incertitudes, E_nominal)

            # Borne A
            x_contact_inc_a = p_inc_a.point_contact(E_bound_a)
            label_to_use_a = label_incert if label_base not in plotted_incert_labels else None
            if x_contact_inc_a is not None and x_contact_inc_a > 0:
                all_x_max_global.append(x_contact_inc_a)
                p_inc_a.tracer_trajectoire(ax, E_bound_a, 0, x_contact_inc_a, color=color, label=label_to_use_a, is_uncertainty_plot=True)
                if label_to_use_a: plotted_incert_labels.add(label_base)
            else:
                non_contact_incert_info.append({'p': p_inc_a, 'E': E_bound_a, 'color': color, 'label': label_to_use_a})

            # Borne B
            x_contact_inc_b = p_inc_b.point_contact(E_bound_b)
            label_to_use_b = None # Jamais de label pour la 2eme borne
            if x_contact_inc_b is not None and x_contact_inc_b > 0:
                all_x_max_global.append(x_contact_inc_b)
                p_inc_b.tracer_trajectoire(ax, E_bound_b, 0, x_contact_inc_b, color=color, label=label_to_use_b, is_uncertainty_plot=True)
            else:
                non_contact_incert_info.append({'p': p_inc_b, 'E': E_bound_b, 'color': color, 'label': label_to_use_b})

        except ValueError as e:
             print(f"Erreur incertitude pour {label_base}: {e}")

    # Finalisation plot
    if all_x_max_global: xlim_max = max(all_x_max_global) * 1.1
    else: xlim_max = hauteur_initiale * 2
    ax.set_xlim(0, xlim_max)

    # Tracer les non-contacts maintenant que xlim est défini
    for item in non_contact_nominal_info:
        item['p'].tracer_trajectoire(ax, E_nominal, 0, xlim_max, color=item['color'], label=f"{item['label']} (pas contact)")
    for item in non_contact_incert_info:
        item['p'].tracer_trajectoire(ax, item['E'], 0, xlim_max, color=item['color'], label=item['label'], is_uncertainty_plot=True)


    ax.plot([0, xlim_max], [0, 0], c='black', linewidth=3, label='Échantillon (y=0)')
    ax.text(0.98, 0.98, texte_angles, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Déviation Électrique (V={potentiel:.1f} V) avec Incertitudes")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()


# --- Fonctions Potentiels Multiples (NON MODIFIÉES - non appelées par l'UI pour incertitudes) ---
# ... (tracer_ensemble_potentiels et tracer_ensemble_trajectoires_potentiels_avec_incertitudes inchangées) ...
def tracer_ensemble_potentiels(
        # ... (code inchangé) ...
        masse_charge_particule : tuple[float, float],
        vitesse_initiale : float,
        potentiels : list[float],
        angle_initial : float, # Radians
        hauteur_initiale : float,
        labels_particules: list[str] = None, # Ignoré ici, un seul type de particule
        create_plot=True,
        ax=None
    ) -> None:
    if create_plot or ax is None : fig, ax = plt.subplots(figsize=(10, 8))

    p = particule(masse_charge_particule, vitesse_initiale, angle_initial, hauteur_initiale)
    texte_angles = "Angles incidents (vs +x):"
    all_x_max = []
    cmap = plt.cm.viridis.resampled(len(potentiels)) # Colormap
    non_contact_list_info = []

    for i, V in enumerate(sorted(potentiels)): # Trier pour ordre couleurs
        color = cmap(i / len(potentiels) if len(potentiels) > 1 else 0.5)
        E = champ_electrique_v2(hauteur_initiale, V)
        x_contact = p.point_contact(E)
        label = f"V = {V:.1f} V"

        if x_contact is not None and x_contact > 0:
            all_x_max.append(x_contact)
            p.tracer_trajectoire(ax, E, 0, x_contact, color=color, label=label)
            angle_inc = p.angle_incident(E) # Angle vs +x
            angle_deg = np.degrees(angle_inc) if angle_inc is not None else None
            texte_angles += f"\n- {V:.0f}V: {angle_deg:.1f}°" if angle_deg is not None else f"\n- {V:.0f}V: Contact?" # Garder tel quel
            is_contact_found = True
        else:
            texte_angles += f"\n- {V:.0f}V: Pas contact (x>0)"
            non_contact_list_info.append({'p':p, 'E':E, 'color':color, 'label':label})


    # Finalisation plot
    if all_x_max: xlim_max = max(all_x_max) * 1.1
    else: xlim_max = hauteur_initiale * 2
    ax.set_xlim(0, xlim_max)

    # Tracer non-contacts
    for item in non_contact_list_info:
        item['p'].tracer_trajectoire(ax, item['E'], 0, xlim_max, color=item['color'], label=f"{item['label']} (pas contact)")
        if xlim_max not in all_x_max: all_x_max.append(xlim_max) # Au cas où TOUS sont non-contact


    ax.plot([0, xlim_max], [0, 0], c='black', linewidth=3, label='Échantillon (y=0)')
    ax.text(0.98, 0.98, texte_angles, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Effet Potentiel sur {p.m:.1f}u, {p.c:+.0f}e")
    ax.legend(title="Potentiel (V)", fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()


def tracer_ensemble_trajectoires_potentiels_avec_incertitudes(
    # ... (code inchangé) ...
    masse_charge_particule : tuple[float, float],
    vitesse_initiale : float,
    incertitudes : dict,
    potentiels : list[float], # Sera seulement [pot1, pot2] lors de l'appel
    angle_initial : float,
    hauteur_initiale : float,
    labels_particules: list[str] = None, # Ignoré
    create_plot=True,
    ax=None
) -> None:
    if create_plot or ax is None: fig, ax = plt.subplots(figsize=(10, 8))

    p_base = particule(masse_charge_particule, vitesse_initiale, angle_initial, hauteur_initiale)
    all_x_max = []
    texte_angles = "Angles incidents (Nominal) (vs +x):"
    plotted_incert_labels = set()
    cmap = plt.cm.viridis.resampled(len(potentiels)) # Colormap pour potentiels
    non_contact_nominal_info = []
    non_contact_incert_info = []

    for i, V in enumerate(sorted(potentiels)):
        color = cmap(i / len(potentiels) if len(potentiels) > 1 else 0.5)
        E_nominal = champ_electrique_v2(hauteur_initiale, V)
        label_base = f"V={V:.1f} V (Nominal)"
        label_incert = f"V={V:.1f} V (Incert.)"

        # Tracer nominal
        x_contact_nom = p_base.point_contact(E_nominal)
        if x_contact_nom is not None and x_contact_nom > 0:
            all_x_max.append(x_contact_nom)
            p_base.tracer_trajectoire(ax, E_nominal, 0, x_contact_nom, color=color, label=label_base)
            angle_inc = p_base.angle_incident(E_nominal)
            angle_deg = np.degrees(angle_inc) if angle_inc is not None else None
            texte_angles += f"\n- {V:.0f}V: {angle_deg:.1f}°" if angle_deg is not None else f"\n- {V:.0f}V: Contact?"
        else:
            texte_angles += f"\n- {V:.0f}V: Pas contact (x>0)"
            non_contact_nominal_info.append({'p':p_base, 'E':E_nominal, 'color':color, 'label':label_base})


        # Créer et tracer incertitudes
        try:
            p_inc1, p_inc2, E_bound1, E_bound2 = create_incertitude_params(p_base, incertitudes, E_nominal)
            # Borne A
            x_contact_inc1 = p_inc1.point_contact(E_bound1)
            label_to_use = label_incert if V not in plotted_incert_labels else None
            if x_contact_inc1 is not None and x_contact_inc1 > 0:
                all_x_max.append(x_contact_inc1)
                p_inc1.tracer_trajectoire(ax, E_bound1, 0, x_contact_inc1, color=color, label=label_to_use, is_uncertainty_plot=True)
                if label_to_use: plotted_incert_labels.add(V)
            else:
                 non_contact_incert_info.append({'p':p_inc1, 'E':E_bound1, 'color':color, 'label':label_to_use})

            # Borne B
            x_contact_inc2 = p_inc2.point_contact(E_bound2)
            label_to_use = None # Jamais de label pour 2eme borne
            if x_contact_inc2 is not None and x_contact_inc2 > 0:
                all_x_max.append(x_contact_inc2)
                p_inc2.tracer_trajectoire(ax, E_bound2, 0, x_contact_inc2, color=color, label=None, is_uncertainty_plot=True)
            else:
                 non_contact_incert_info.append({'p':p_inc2, 'E':E_bound2, 'color':color, 'label':None})

        except ValueError as e:
             print(f"Erreur incertitude pour V={V}V: {e}")

    # Finalisation
    if all_x_max: xlim_max = max(all_x_max) * 1.1
    else: xlim_max = hauteur_initiale * 2
    ax.set_xlim(0, xlim_max)

    # Tracer non-contacts
    for item in non_contact_nominal_info: item['p'].tracer_trajectoire(ax, item['E'], 0, xlim_max, color=item['color'], label=f"{item['label']} (pas contact)")
    for item in non_contact_incert_info: item['p'].tracer_trajectoire(ax, item['E'], 0, xlim_max, color=item['color'], label=item['label'], is_uncertainty_plot=True)


    ax.plot([0, xlim_max], [0, 0], c='black', linewidth=3, label='Échantillon (y=0)')
    ax.text(0.98, 0.98, texte_angles, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Effet Potentiel sur {p_base.m:.1f}u, {p_base.c:+.0f}e (avec Incertitudes)")
    ax.legend(title="Potentiel (V) / Courbe", fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()



# --- Bloc Test ---
if __name__ == '__main__':
    # ... (tests précédents) ...
    pass # Plus de tests par défaut