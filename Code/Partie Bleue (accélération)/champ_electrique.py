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
    

    # Niveau 5.1 : Equations temporelles de la particule dans un champ constant donné
    def equations_temporelles(self, t : float, Ey : float, Ex : float = 0, Ez : float = 0) -> tuple[float, float, float] :    
        """
        Les équations temporelles en x, y et z

        Parameters
        ----------
        t : float
            Temps (en s)
        Ex : float
            Valeur du champ électrique d'axe x (en V/m)
        Ey : float
            Valeur du champ électrique d'axe y (en V/m)
        Ez : float
            Valeur du champ électrique d'axe z (en V/m)

        Returns
        -------
        tuple of (float, float, float)
            - Position en x
            - Position en y
            - Position en z
        """             
        x = Ex * t * t * 0.5 / self.mq
        y = Ey * t * t * 0.5 / self.mq  + self.vo
        z = Ez * t * t * 0.5 / self.mq

        return x, y, z
    
    
    # Niveau 5.2 : Fonction représentant l’équation de la vitesse en fonction de la position d’un ion accéléré par un champ électrique connu.
    def equation_vitesse_fct_position(self, y_pos : float, Ey : float) -> float :
        """
        L'équation de la vitesse selon l'axe y par rapport à la position en y

        Parameters
        ----------
        y_pos : float
            Position en y (en m)
        Ey : float
            Valeur du champ électrique d'axe y (en V/m)

        Returns
        -------
        float
            Vitesse (en m/s)
        """
        return np.sqrt(2 * Ey * (y_pos - self.vo) / self.mq) + self.vo
    

    # Niveau 4.1 : Fonction qui calcule l'évolution de la position des ions accélérés sur une échelle de temps
    def position(self, Ey : float, t_max : float, t_min : float = 0, n_points : int = 10000, Ex : float = 0, Ez : float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la position entre un temps de début et un temps final dans un champ donné

        Parameters
        ----------
        Ey : float
            Valeur du champ électrique d'axe y (en V/m)
        t_max : float
            Temps final (en s)
        t_min : float
            Temps de début (en s)   
        n_points : int
            Nombre de points où la position sera calculée entre t_min et t_max
        Ex : float
            Valeur du champ électrique d'axe x (en V/m)
        Ez : float
            Valeur du champ électrique d'axe z (en V/m)
            
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            - Valeurs de temps
            - Positions en x
            - Positions en y
            - Positions en z
        """
        t = np.linspace(t_min, t_max, n_points)
        x, y, z = self.equations_temporelles(t, Ey, Ex, Ez)
        return t, x, y, z
    

    # Niveau 4.2 : Fonction qui calcule l'évolution de la vitesse par rapport à la position 
    def vitesse_fct_y(self, Ey : float, y_min : float, y_max : float, n_points : float) -> tuple[np.ndarray, np.ndarray] :
        """
        Calcule la vitesse en y entre un y minimum et un y maximum

        Parameters
        ----------
        Ey : float
            Valeur du champ électrique d'axe y (en V/m)
        y_min : float
            Position en y minimale (en m)
        y_max : float
            Position en y maximale (en m)   
        n_points : int
            Nombre de points où la position sera calculée entre y_min et y_max
        
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            - Positions en y
            - Vitesses en y
        """
        y = np.linspace(y_min, y_max, n_points)
        return y, self.equation_vitesse_fct_position(y, Ey)



# # Test fonction position
# if __name__ == '__main__' :
#     p = particule(1e-27/1.602e-19, 0)
#     print(p.position(1, 2))