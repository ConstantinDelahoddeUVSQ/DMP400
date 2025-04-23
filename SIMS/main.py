"""
Fichier qui rassemble toutes les fonctionailités graphiques et 1 interface
"""

import sys
import tkinter as tk
from tkinter import ttk

# Permettre l'import des fichiers
sys.path.append("./SIMS/Partie Bleue (accélération)/Code")
sys.path.append("./SIMS/Partie Verte (déviation magnétique)/Code")

import deviation, partie_electroaimant  # type: ignore


import tkinter as tk
from tkinter import ttk


# ChatGPT made (à modifer)
class ParticleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Deviation Design App")

        self.particles = []

        # Section: Particle Entry
        particle_frame = ttk.LabelFrame(root, text="Particules")
        particle_frame.pack(fill="x", padx=10, pady=10)

        self.particle_container = tk.Frame(particle_frame)
        self.particle_container.pack()

        self.add_particle_row()

        add_particle_btn = ttk.Button(particle_frame, text="Ajouter une particule", command=self.add_particle_row)
        add_particle_btn.pack(pady=5)

        # Section: Déviation magnétique
        magnetic_frame = ttk.LabelFrame(root, text="Déviation magnétique")
        magnetic_frame.pack(fill="x", padx=10, pady=10)

        self.v_init_min = self.add_labeled_entry(magnetic_frame, "Vitesse initiale min")
        self.v_init_max = self.add_labeled_entry(magnetic_frame, "Vitesse initiale max")
        self.champ_min = self.add_labeled_entry(magnetic_frame, "Champ magnétique minimal")
        self.champ_max = self.add_labeled_entry(magnetic_frame, "Champ magnétique maximal")
        self.x_detecteur = self.add_labeled_entry(magnetic_frame, "X_détecteur")

        ttk.Button(magnetic_frame, text="Tracer", command=self.tracer_electroaimant).pack(pady=5)

        # Section: Déviation plaque
        plaque_frame = ttk.LabelFrame(root, text="Déviation plaque")
        plaque_frame.pack(fill="x", padx=10, pady=10)

        self.vitesse_plaque = self.add_labeled_entry(plaque_frame, "Vitesse initiale")

        ttk.Button(plaque_frame, text="Tracer", command=self.tracer_deviation).pack(pady=5)

    def add_labeled_entry(self, parent, label_text):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=2)
        label = ttk.Label(frame, text=label_text, width=25, anchor="w")
        label.pack(side="left")
        entry = ttk.Entry(frame)
        entry.pack(side="left", expand=True, fill="x")
        return entry

    def add_particle_row(self):
        row_frame = tk.Frame(self.particle_container)
        row_frame.pack(fill="x", pady=2)

        charge_entry = ttk.Entry(row_frame, width=15)
        charge_entry.insert(0, "Charge")
        charge_entry.pack(side="left", padx=5)

        mass_entry = ttk.Entry(row_frame, width=15)
        mass_entry.insert(0, "Masse")
        mass_entry.pack(side="left", padx=5)

        self.particles.append((charge_entry, mass_entry))

    def get_particles(self):
        result = []
        for charge_entry, mass_entry in self.particles:
            try:
                charge = float(charge_entry.get())
                mass = float(mass_entry.get())
                result.append({'charge': charge, 'mass': mass})
            except ValueError:
                continue
        return result

    def tracer_electroaimant(self):
        particules = self.get_particles()
        try:
            vmin = float(self.v_init_min.get())
            vmax = float(self.v_init_max.get())
            bmin = float(self.champ_min.get())
            bmax = float(self.champ_max.get())
            x_det = float(self.x_detecteur.get())
        except ValueError:
            print("Erreur dans les entrées numériques.")
            return

        print("Tracer électroaimant avec:")
        print(particules, vmin, vmax, bmin, bmax, x_det)
        # Replace with your actual function
        # tracer_electroaimant(particules, vmin, vmax, bmin, bmax, x_det)

    def tracer_deviation(self):
        particules = self.get_particles()
        try:
            vitesse = float(self.vitesse_plaque.get())
        except ValueError:
            print("Erreur dans l'entrée de vitesse.")
            return

        print("Tracer déviation plaque avec:")
        print(particules, vitesse)
        # Replace with your actual function
        # tracer_deviation(particules, vitesse)

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()



def main() :
    pass




# if __name__ == '__main__' : 
#     main()