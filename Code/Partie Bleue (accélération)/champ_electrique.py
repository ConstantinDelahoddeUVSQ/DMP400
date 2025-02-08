# Objectif 2

import matplotlib.pyplot as plt
import numpy as np


class particule :
    def __init__(self, rapport_masse_charge : float, v_initiale : float = 0) :
        """
        Objet particule accéléré dans un champ électrique d'axe y

        Parameters
        ----------
        rapport_masse_charge : float
            Masse (en Kg) / Charge (en C) de la particule
        v_initiale : float
            Vitesse initiale en y de la particule (en m/s)   
        """
        self.mq = rapport_masse_charge
        self.vo = v_initiale
    
    
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
        prefix = self.mq / Bz
        return self.vo * prefix * np.sin(np.arccos(1 - x / (self.vo * prefix)))