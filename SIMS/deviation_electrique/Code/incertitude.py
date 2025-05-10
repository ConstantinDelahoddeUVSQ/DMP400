import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

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

if __name__ == "__main__":
    # Paramètres communs pour le tracé
    v0        = 2e5       # m/s
    theta     = np.pi/6
    y0        = 0.05      # m
    distance  = 0.05      # m
    delta_V   = -5000     # V

    # Générer 1 000 masses de 1 à 10 u
    m_u_list = np.linspace(1, 10, 1000)
    q_e_list = np.ones_like(m_u_list)

    E = champ_electrique_v2(distance, delta_V)

    mq_vals = np.empty_like(m_u_list)
    inc_vals = np.empty_like(m_u_list)

    for i, (m_u, q_e) in enumerate(zip(m_u_list, q_e_list)):
        m_si = m_u * constants.u
        q_si = q_e * constants.e
        mq_vals[i] = m_si / q_si
        deltas = (
            v0*0.01,        # Δv0 = 1%
            theta*0.02,     # Δθ = 2%
            y0*0.05,        # Δy0 = 5%
            q_si*0.001,     # Δq = 0.1%
            m_si*0.001,     # Δm = 0.1%
            abs(E)*0.03     # ΔE = 3%
        )
        inc_vals[i] = calculer_incertitude(v0, theta, y0, q_si, m_si, E, *deltas)

    # Tracé final
    plt.figure(figsize=(8,5))
    plt.plot(mq_vals, inc_vals)
    plt.xscale('log')
    plt.xlabel("m/q (kg/C)")
    plt.ylabel("Δ xs (m)")
    plt.title("Incertitude sur xs avec 1 000 points (1–10 u)")
    plt.show()