import numpy as np

def calculer_xs(v0: float, theta: float, y0: float, q: float, m: float, E: float) -> float:
    """
    Calcule l'abscisse de contact xs selon la formule dérivée.

    Parameters
    ----------
    v0 : float
        Vitesse initiale de la particule (en m/s)
    theta : float
        Angle initial entre la vitesse et l'axe y (en radians)
    y0 : float
        Hauteur initiale de la particule (en m)
    q : float
        Charge de la particule (en C)
    m : float
        Masse de la particule (en kg)
    E : float
        Intensité du champ électrique uniforme selon l'axe y (en V/m)

    Returns
    -------
    float
        L'abscisse x (xs) où la particule atteint y=0 (en m)

    Raises
    ------
    ValueError
        Si le discriminant est négatif (pas de contact avec l'axe y=0).
    """
    A = v0 * np.cos(theta)
    B = 2 * y0 * q * E / m
    C = q * E / m
    
    discriminant = A**2 - B
    if discriminant < 0:
        raise ValueError("Le discriminant est négatif (A^2 < B), pas de solution réelle pour xs (pas de contact)")
    D = np.sqrt(discriminant)

    xs = ((A - D) / C) * v0 * np.sin(theta)
    return xs

def derivees_partielles(v0: float, theta: float, y0: float, q: float, m: float, E: float) -> tuple[float, float, float, float, float, float]:
    """
    Calcule les dérivées partielles de xs par rapport à chaque variable d'entrée.

    Parameters
    ----------
    v0 : float
        Vitesse initiale de la particule (en m/s)
    theta : float
        Angle initial entre la vitesse et l'axe y (en radians)
    y0 : float
        Hauteur initiale de la particule (en m)
    q : float
        Charge de la particule (en C)
    m : float
        Masse de la particule (en kg)
    E : float
        Intensité du champ électrique uniforme selon l'axe y (en V/m)

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Un tuple contenant les dérivées partielles dans l'ordre :
        (dxs/dv0, dxs/dtheta, dxs/dy0, dxs/dq, dxs/dm, dxs/dE)

    Raises
    ------
    ValueError
        Si le discriminant est négatif ou nul (division par zéro dans D ou dérivée non définie).
    """
    A = v0 * np.cos(theta)
    B = 2 * y0 * q * E / m
    C = q * E / m
    D = np.sqrt(A**2 - B)
    
    # Dérivée par rapport à v0
    dxs_dv0 = (np.sin(theta) / C) * (A - D) + (v0 * np.sin(theta) * np.cos(theta) / C) * (1 - A / D)
    
    # Dérivée par rapport à theta
    dxs_dtheta = (v0 * np.cos(theta) / C) * (A - D) - (v0**2 * np.sin(theta)**2 / C) * (1 - A / D)
    
    # Dérivée par rapport à y0
    dxs_dy0 = -v0 * np.sin(theta) * q * E / (C * m * D)
    
    # Dérivée par rapport à q
    dxs_dq = -(A - D) * E * v0 * np.sin(theta) / (C**2 * m) - v0 * np.sin(theta) * y0 * E / (C * m * D)
    
    # Dérivée par rapport à m
    dxs_dm = (A - D) * q * E * v0 * np.sin(theta) / (C**2 * m**2) + v0 * np.sin(theta) * y0 * q * E / (C * m**2 * D)
    
    # Dérivée par rapport à E
    dxs_dE = -(A - D) * q * v0 * np.sin(theta) / (C**2 * m) - v0 * np.sin(theta) * y0 * q / (C * m * D)
    
    return dxs_dv0, dxs_dtheta, dxs_dy0, dxs_dq, dxs_dm, dxs_dE

def calculer_incertitude(v0: float, theta: float, y0: float, q: float, m: float, E: float,
                           delta_v0: float, delta_theta: float, delta_y0: float,
                           delta_q: float, delta_m: float, delta_E: float) -> float:
    """
    Calcule l'incertitude totale sur xs en utilisant la propagation des erreurs
    par les dérivées partielles (méthode de première ordre).

    Parameters
    ----------
    v0 : float
        Valeur de la vitesse initiale (en m/s)
    theta : float
        Valeur de l'angle initial (en radians)
    y0 : float
        Valeur de la hauteur initiale (en m)
    q : float
        Valeur de la charge (en C)
    m : float
        Valeur de la masse (en kg)
    E : float
        Valeur du champ électrique (en V/m)
    delta_v0 : float
        Incertitude sur la vitesse initiale (en m/s)
    delta_theta : float
        Incertitude sur l'angle initial (en radians)
    delta_y0 : float
        Incertitude sur la hauteur initiale (en m)
    delta_q : float
        Incertitude sur la charge (en C)
    delta_m : float
        Incertitude sur la masse (en kg)
    delta_E : float
        Incertitude sur le champ électrique (en V/m)

    Returns
    -------
    float
        L'incertitude absolue calculée sur xs (Δxs) (en m)

    Raises
    ------
    ValueError
        Si les dérivées partielles ne peuvent être calculées (discriminant <= 0).
    """
    dxs_dv0, dxs_dtheta, dxs_dy0, dxs_dq, dxs_dm, dxs_dE = derivees_partielles(v0, theta, y0, q, m, E)
    
    # Calcul de l'incertitude totale
    delta_xs = np.sqrt(
        (dxs_dv0 * delta_v0)**2 +
        (dxs_dtheta * delta_theta)**2 +
        (dxs_dy0 * delta_y0)**2 +
        (dxs_dq * delta_q)**2 +
        (dxs_dm * delta_m)**2 +
        (dxs_dE * delta_E)**2
    )
    
    return delta_xs

# Exemple d'utilisation du code
if __name__ == "__main__":
    # Valeurs par défaut pour les paramètres
    v0 = 1e6  # m/s
    theta = np.pi/4  # rad
    y0 = 1e-2  # m
    q = 1.602e-19  # C
    m = 9.109e-31  # kg
    E = 1e3  # V/m
    
    # Incertitudes sur les paramètres (exemple)
    delta_v0 = v0 * 0.01  # 1% d'incertitude sur v0
    delta_theta = theta * 0.02  # 2% d'incertitude sur theta
    delta_y0 = y0 * 0.05  # 5% d'incertitude sur y0
    delta_q = q * 0.001  # 0.1% d'incertitude sur q (généralement très bien connu)
    delta_m = m * 0.001  # 0.1% d'incertitude sur m (généralement très bien connu)
    delta_E = E * 0.03  # 3% d'incertitude sur E