import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants


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
        """
        Objet particule avec vitesse initiale dévié par un champ électrique d'axe y

        Parameters
        ----------
        masse_charge : tuple of float
            Masse (en u) / Charge (nombre de charge élémentaire) de la particule
        v_initiale : float
            Vitesse initiale de la particule (en m/s)   
        angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
        hauteur_initiale : float
            Coordonnée en y du point de départ
        is_incertitude : bool
            Permet de savoir si la particule est une incertitude qui sera tracée
        incertitude_unique : bool
            Permet de savoir si cette particule représente la première ou deuxième incertitude (pour ne pas donner le label 2 fois)
        base_mq : tuple 
            Si la particule est une particule incertitude permet d'avoir le couple masse charge de la vraie particule d'origine
        """
        mass_u, charge_e = masse_charge
        if charge_e == 0: raise ValueError("La charge ne peut pas être nulle.")
        if mass_u <= 0: raise ValueError("La masse doit être positive.")
        if v_initiale < 0: raise ValueError("La vitesse initiale ne peut être négative.")
        if not (0 < angle_initial < np.pi/2): raise ValueError("L'angle initial doit être entre 0 et pi/2 radians (exclus).")
        if hauteur_initiale <= 0: raise ValueError("La hauteur initiale doit être positive.")

        self.mq = (mass_u * constants.u) / (charge_e * constants.e)
        self.vo = v_initiale
        self.angle = angle_initial 
        self.height = hauteur_initiale
        self.m = mass_u
        self.c = charge_e
        self.is_incertitude = is_incertitude
        self.incertitude_unique = incertitude_unique
        self.base_mq = base_mq if base_mq else (mass_u, charge_e)


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
                    return self.height * np.tan(self.angle)
            else :
                return None

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

    def tracer_trajectoire(self, ax, E : float, x_min : float, x_max : float, color=None, label=None, is_uncertainty_plot=False, n_points : int = 1000) -> None:
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
        color : str or list
            Couleur du tracé pour les incertitudes
        label : str
            Label du tracé pour les incertitudes
        n_points : int
            Nombre de points où la position sera calculée entre x_min et x_max
        """
        x, y = self.trajectoire(E, x_min, x_max, n_points)
        if len(x) == 0: return
        plot_kwargs = {}
        plot_kwargs['c'] = color
        plot_kwargs['label'] = label
        if is_uncertainty_plot:
            plot_kwargs['linestyle'] = '--'; plot_kwargs['alpha'] = 0.7

        ax.plot(x, y, **plot_kwargs)


def calculer_trajectoire_et_impact(
    masse_charge: tuple[float, float],
    vitesse_initiale: float,
    potentiel: float,
    angle_initial_rad: float,
    hauteur_initiale: float,
    n_points: int = 1000
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Fonction utilisée pour la variation de potentiel.
    Calcule les trajectoires et le x_impact

    Parameters
    ----------
    masse_charge : tuple of float
        Masse (en unités atomiques), Charge (nombre de charge élémentaire) de la particule
    vitesse_initiale : float
        Vitesse intiale de la particule
    potentiel : float
        Différence de potentiel entre les plaques (en V)
    angle_initial_rad : float
        Angle initial entre v_initiale et l'axe y en radians 
    hauteur_initiale : float
        Coordonnée en y du point de départ
    n_points : int
        Nombre de points pour calculer la trajectoire
    """
    try:
        p = particule(masse_charge, vitesse_initiale, angle_initial_rad, hauteur_initiale)
        E = champ_electrique_v2(hauteur_initiale, potentiel)
        x_impact = p.point_contact(E) 

        if x_impact is not None and x_impact > 0:
            x_traj, y_traj = p.trajectoire(E, 0, x_impact, n_points) 
        else:
            x_max_plot = max(hauteur_initiale * np.tan(angle_initial_rad) * 2, 0.1) if np.tan(angle_initial_rad)!=0 else 0.1
            x_traj, y_traj = p.trajectoire(E, 0, x_max_plot, n_points)
            x_impact = None

        return x_traj, y_traj, x_impact
    except ValueError as e:
        print(f"Erreur calcul traj.: {e}")
        return np.array([]), np.array([]), None

def tracer_ensemble_trajectoires(
        masse_charge_particules : list[tuple[float, float]],
        vitesse_initiale : float,
        potentiel : float,
        angle_initial : float, # Radians
        hauteur_initiale : float,
        labels_particules: list[str] = None, # Liste des noms
        create_plot=True,
        ax=None
    ) -> None :
    """
    Trace les trajectoires jusqu'au contact de différentes particules de manière statique

    Parameters
    ----------
    masse_charge_particules : list of tuple of float
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale commune à toutes les particules du faisceau
    potentiel : float
        Différence de potentiel entre les plaques (en V)
    angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ
    labels_particules : list of str
        Liste des labels pour toutes les particules
    create_plot : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.
    ax : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.
    """
    if create_plot or ax is None : fig, ax = plt.subplots(figsize=(10, 8))
    if labels_particules is None: labels_particules = [f"Particule {i+1}" for i in range(len(masse_charge_particules))]
    if len(labels_particules) != len(masse_charge_particules):
        print("Avertissement: Noms/Particules mismatch.")
        labels_particules = [f"{mc[0]:.1f}u,{mc[1]:+.0f}e" for mc in masse_charge_particules] # Fallback labels

    E = champ_electrique_v2(hauteur_initiale, potentiel)
    all_x_max = []
    texte_angles = "Angles incidents (vs +x):"
    is_contact = False
    non_contact_list_info = [] 

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


def create_incertitude_params(p : particule, incertitudes : dict, E : float) :
    """
    Crée une des particules min et max 'incertitude' et les E_min, E_max

    Parameters
    ----------
    p : particule
        objects particule
    incertitudes : dict
        Dictionnaire des incertitudes de chaque paramètre (en pourcentages)
    E : float
        Champ électrique (T)
    """
    if p.c * E >= 0 :
            min_particule = particule((p.m * (1 + incertitudes['m']), p.c * (1 - incertitudes['q'])), p.vo * (1 + incertitudes['v0']), p.angle * (1 - incertitudes['theta']), p.height * (1 - incertitudes['h']), is_incertitude=True, incertitude_unique = True, base_mq=(p.m, p.c))
            max_particule = particule((p.m * (1 - incertitudes['m']), p.c * (1 + incertitudes['q'])), p.vo * (1 - incertitudes['v0']), p.angle * (1 + incertitudes['theta']), p.height * (1 + incertitudes['h']), is_incertitude=True, base_mq=(p.m, p.c))
            E_min = E * (1 - incertitudes['E'])
            E_max = E * (1 + incertitudes['E'])
    else :
            min_particule = particule((p.m * (1 - incertitudes['m']), p.c * (1 + incertitudes['q'])), p.vo * (1 - incertitudes['v0']), p.angle * (1 - incertitudes['theta']), p.height * (1 - incertitudes['h']), is_incertitude=True, incertitude_unique = True, base_mq=(p.m, p.c))
            max_particule = particule((p.m * (1 + incertitudes['m']), p.c * (1 - incertitudes['q'])), p.vo * (1 + incertitudes['v0']), p.angle * (1 + incertitudes['theta']), p.height * (1 + incertitudes['h']), is_incertitude=True, base_mq=(p.m, p.c))
            E_min = E * (1 + incertitudes['E'])
            E_max = E * (1 - incertitudes['E'])

    return min_particule, max_particule, E_min, E_max

def tracer_ensemble_trajectoires_avec_incertitudes(
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
    """
    Trace les trajectoires jusqu'au contact de différentes particules de manière statique avec le tracé des incertitudes (couloirs)

    Parameters
    ----------
    masse_charge_particules : list of tupleof int
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale commune à toutes les particules du faisceau
    incertitudes : dict
        Dictionnaire des incertitudes sur les différents paramètres (pourcentages)
    potentiel : float
        Différence de potentiel entre les plaques (en V)
    angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ
    labels_particules : list of str
        Liste des labels pour toutes les particules
    create_plot : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.
    ax : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.

    """
    if create_plot or ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    if labels_particules is None: labels_particules = [f"P{i+1}" for i in range(len(masse_charge_particules))]
    if len(labels_particules) != len(masse_charge_particules): labels_particules = [f"{mc[0]:.1f}u,{mc[1]:+.0f}e" for mc in masse_charge_particules]

    E_nominal = champ_electrique_v2(hauteur_initiale, potentiel)
    particules_base = [particule(mc, vitesse_initiale, angle_initial, hauteur_initiale) for mc in masse_charge_particules]

    all_x_max_global = []
    texte_angles = "Angles incidents (Nominal) (vs +x):"
    plotted_incert_labels = set()
    colors = plt.cm.viridis(np.linspace(0, 1, len(particules_base)))
    non_contact_nominal_info = [] 
    non_contact_incert_info = [] 

    for i, p_base in enumerate(particules_base):
        color = colors[i]; label_base = labels_particules[i]
        label_incert = f"Incert. {label_base}"

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


def calculer_delta_impact(masse_charge_tuple : tuple, vitesse_initiale : float, potentiel_ref : float, potentiel : float, angle_initial_rad : float, hauteur_initiale : float) -> float :
    """
    Fonction qui calcule l'écart de point de contact pour une particule entre 2 potentiels

    Parameters
    ----------
    masse_charge_tuple : tuple of float
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour la particule
    vitesse_initiale : float
        Vitesse intiale
    potentiel_ref : float
        Différence de potentiel entre les plaques (en V) référence
    potentiel : float
        Différence de potentiel entre les plaques (en V) référence
    angle_initial_rad : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ
    """
    p = particule(masse_charge_tuple, vitesse_initiale, angle_initial_rad, hauteur_initiale)
    xs_ref = p.point_contact(champ_electrique_v2(hauteur_initiale, potentiel_ref))
    xs = p.point_contact(champ_electrique_v2(hauteur_initiale, potentiel))
    return xs - xs_ref


def tracer_ensemble_potentiels(
        masse_charge_particule : tuple[float, float],
        vitesse_initiale : float,
        potentiels : list[float],
        angle_initial : float, # Radians
        hauteur_initiale : float,
        labels_particules: list[str] = None, # Ignoré ici, un seul type de particule
        create_plot=True,
        ax=None
    ) -> None:
    """
    Trace les trajectoires jusqu'au contact d'une particule pour plusieurs potentiels de manière statique

    Parameters
    ----------
    masse_charge_particule : tuple of float
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour la particule
    vitesse_initiale : float
        Vitesse intiale commune à toutes les particules du faisceau
    potentiels : list of float
        Différence de potentiel entre les plaques (en V)
    angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ
    create_plot : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.
    ax : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.

    """
    if create_plot or ax is None : fig, ax = plt.subplots(figsize=(10, 8))

    p = particule(masse_charge_particule, vitesse_initiale, angle_initial, hauteur_initiale)
    texte_angles = "Angles incidents (vs +x):"
    all_x_max = []
    cmap = plt.cm.viridis # Colormap
    non_contact_list_info = []

    for i, V in enumerate(sorted(potentiels)): # Trier pour ordre couleurs
        if len(potentiels) > 1 :
            color = cmap(i / len(potentiels))
        else :
            color = cmap(0.5)
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

    delta_xs = calculer_delta_impact(masse_charge_particule, vitesse_initiale, potentiels[0], potentiels[1], angle_initial, hauteur_initiale)
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Effet Potentiel sur {p.m:.1f}u, {p.c:+.0f}e\n Delta Xs : {delta_xs:+.3e}")
    ax.legend(title="Potentiel (V)", fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()


def tracer_ensemble_trajectoires_potentiels_avec_incertitudes(
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
    """
    Trace les trajectoires jusqu'au contact de différentes particules de manière statique avec le tracé des incertitudes (couloirs)

    Parameters
    ----------
    masse_charge_particules : tuple of float
        Masse (en unités atomiques), Charge (nombre de charge élémentaire)  pour toutes les particules
    vitesse_initiale : float
        Vitesse intiale commune à toutes les particules du faisceau
    incertitudes : dict
        Dictionnaire des incertitudes sur les différents paramètres (pourcentages)
    potentiels : float
        Différence de potentiel entre les plaques (en V)
    angle_initial : float
            Angle initial entre v_initiale et l'axe y en radians
    hauteur_initiale : float
        Coordonnée en y du point de départ
    create_plot : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.
    ax : bool
        Permet de maneuvrer la meme fonction pour l'utilisateur et l'interface.

    """
    if create_plot or ax is None: fig, ax = plt.subplots(figsize=(10, 8))

    p_base = particule(masse_charge_particule, vitesse_initiale, angle_initial, hauteur_initiale)
    all_x_max = []
    texte_angles = "Angles incidents (Nominal) (vs +x):"
    plotted_incert_labels = set()
    cmap = plt.cm.viridis # Colormap pour potentiels
    non_contact_nominal_info = []
    non_contact_incert_info = []

    for i, V in enumerate(sorted(potentiels)):
        if len(potentiels) > 1 :
            color = cmap(i / len(potentiels))
        else :
            color = cmap(0.5)
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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    delta_xs = calculer_delta_impact(masse_charge_particule, vitesse_initiale, potentiels[0], potentiels[1], angle_initial, hauteur_initiale)
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.set_title(f"Effet Potentiel sur {p_base.m:.1f}u, {p_base.c:+.0f}e (avec Incertitudes) \n delta Xs : {delta_xs:+.3e}")
    ax.legend(title="Potentiel (V) / Courbe", fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    if create_plot : plt.show()




"""
Test fonction tracer_ensemble_trajectoires
"""
if __name__ == '__main__' :
    rapports_mq, vo = [(1, 1), (2, 1), (3, 1)], 1e6
    potentiel = 5000
    h_initiale = 0.1
    angle_initial = np.pi / 6

    tracer_ensemble_trajectoires(rapports_mq, vo, potentiel=potentiel, hauteur_initiale=h_initiale, labels_particules=["P1", "P2", "P3"])


"""
Test fonction tracer_ensemble_trajectoires_avec_incertitudes
"""
# if __name__ == '__main__' :
#     rapports_mq, vo = [(1, 1), (3, 1)], 1e6
#     potentiel = 5000
#     h_initiale = 0.1
#     incertitudes = {'m' : 0.001, 'v0' : 0.01, 'theta' : 0.02, 'h' : 0.05, 'q' : 0.001, 'E' : 0.03}


#     tracer_ensemble_trajectoires_avec_incertitudes(rapports_mq, vo, incertitudes, potentiel=potentiel, hauteur_initiale=h_initiale)


"""
Test fonction tracer_ensemble_trajectoires_potentiels_avec_incertitudes
"""
# if __name__ == '__main__' :
#     rapports_mq, vo = (1, 1), 1e5
#     potentiels = [0, 50]
#     h_initiale = 0.1
#     incertitudes = {'m' : 0.001, 'v0' : 0.01, 'theta' : 0.02, 'h' : 0.05, 'q' : 0.001, 'E' : 0.03}


#     tracer_ensemble_trajectoires_potentiels_avec_incertitudes(rapports_mq, vo, incertitudes, potentiels=potentiels, hauteur_initiale=h_initiale)


"""
Test fonction tracer_ensemble_trajectoires_dynamique
"""
# if __name__ == '__main__' :
#     rapports_mq, vo = [(1, 1), (2, 1), (3, 1)], 1e6
#     pot_min, pot_max = -5000, 5000
#     h_initiale = 0.1


#     tracer_ensemble_trajectoires_dynamique(rapports_mq, vo, potentiel_min=pot_min, potentiel_max=pot_max, hauteur_initiale=h_initiale)