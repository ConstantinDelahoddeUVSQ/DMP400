import numpy as np
import scipy.constants as constants

def calculate_xs_formula(v0, theta, y0, q, m, E, x0=0):
  """
  Traduit directement la formule mathématique fournie en code Python.

  xs = [ (v0*cos(theta) - sqrt((v0*cos(theta))^2 - 2*y0*q*E/m)) / (q*E/m) ] * v0*sin(theta) + x0

  Attention : Cette fonction ne gère pas explicitement les cas limites
              (E=0, q=0, racine négative) qui mèneraient à des erreurs
              ou des résultats incorrects (NaN, Inf). Voir la version
              plus robuste pour une utilisation générale.

  Parameters
  ----------
  v0 : float
      Vitesse initiale (m/s).
  theta : float
      Angle initial par rapport à l'axe +y (vertical) (radians).
  y0 : float
      Position verticale initiale (m).
  q : float
      Charge de la particule (Coulombs).
  m : float
      Masse de la particule (kg).
  E : float
      Composante verticale du champ électrique (V/m).
  x0 : float, optional
      Position horizontale initiale (m). Défaut = 0.

  Returns
  -------
  float
      Position horizontale xs calculée selon la formule.
      Peut retourner NaN ou Inf si les conditions mathématiques ne sont pas remplies.
  """

  cos_theta = np.cos(theta)
  sin_theta = np.sin(theta)
  v0_cos_theta = v0 * cos_theta

  with np.errstate(invalid='ignore'):
      term_inside_sqrt = v0_cos_theta**2 - (2 * y0 * q * E) / m
      sqrt_term = np.sqrt(term_inside_sqrt)


  numerator_factor = v0_cos_theta - sqrt_term

  with np.errstate(divide='ignore', invalid='ignore'):
      denominator_factor = (q / m) * E
      time_factor = numerator_factor / denominator_factor

  # Calcul final de xs
  xs = time_factor * v0 * sin_theta + x0

  return xs

# Test 
v0_test = 1e5               
theta_test = np.radians(45)  
y0_test = 0.1               
q_test = constants.e         
m_test = constants.proton_mass 
x0_test = 0.0               

E_test = -50000            

xs_result = calculate_xs_formula(v0_test, theta_test, y0_test, q_test, m_test, E_test, x0_test)

print(f"Paramètres de test:")
print(f"  v0    = {v0_test:.2e} m/s")
print(f"  theta = {np.degrees(theta_test):.1f} degrés (vs +y)")
print(f"  y0    = {y0_test} m")
print(f"  q     = {q_test:.2e} C")
print(f"  m     = {m_test:.2e} kg")
print(f"  E     = {E_test:.2e} V/m")
print(f"  x0    = {x0_test} m")
print("-" * 20)
print(f"Résultat xs = {xs_result} m")
