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
    def equations_temporelles(self, t : float, Ey : float, Ex : float = 0, Ez : float = 0) -> dict[str, float] :    
        """
        Les équations temporelles en x, y et z

        Parameters
        ----------
        t : float
            Temps (en s)
        Ex : float
            Champ électrique selon l'axe x (en V/m)
        Ey : float
            Champ électrique selon l'axe y (en V/m)
        Ez : float
            Champ électrique selon l'axe z (en V/m)

        Returns
        -------
        dict of str to float
            Dictionnaire où :
            - Les clés sont 'x', 'y', 'z' 
            - Les valeurs sont la position sur l'axe donné en clé
        """             
        x = Ex * t * t * 0.5 / self.mq
        y = Ey * t * t * 0.5 / self.mq  + self.vo
        z = Ez * t * t * 0.5 / self.mq

        return {'x' : x, 'y' : y, 'z' : z}
    
    
    # Niveau 5.2 : fonction représentant l’équation de la vitesse en fonction de la position d’un ion accéléré par un champ électrique connu.
    def equation_vitesse_fct_position(self, y_pos : float, Ey : float) -> float :
        """
        L'équation de la vitesse selon l'axe y par rapport à la position en y

        Parameters
        ----------
        y_pos : float
            La position en y (en m)
        Ey : float
            Champ électrique selon l'axe y (en V/m)

        Returns
        -------
        float
            Vitesse (en m/s)
        """
        return np.sqrt(2 * Ey * (y_pos - self.vo) / self.mq) + self.vo