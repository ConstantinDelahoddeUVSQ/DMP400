import sys
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Permettre l'import des fichiers (ajustez les chemins si nécessaire)
# Suppose que le script main.py est dans le dossier parent de SIMS
sys.path.append("./SIMS/Partie Bleue (accélération)/Code")
sys.path.append("./SIMS/Partie Verte (déviation magnétique)/Code")

try:
    import deviation # type: ignore
    import partie_electroaimant # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que les chemins sys.path sont corrects et que les fichiers existent.")
    sys.exit(1)


class ParticleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulateur SIMS - Déviations")
        self.root.geometry("1400x800") # Taille initiale

        # Style
        style = ttk.Style()
        style.theme_use('clam') # Ou 'alt', 'default', 'classic'
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabelframe.Label", font=('Helvetica', 12, 'bold'))
        style.configure("TLabel", padding=2)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))

        # Liste pour stocker les données des particules (mass_u: float, charge_e: float)
        self.particles_data = []

        # --- Structure principale ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Panneau de contrôle (gauche) ---
        control_panel = ttk.Frame(main_paned_window, width=350)
        main_paned_window.add(control_panel, weight=1)

        # --- Section Particules ---
        particle_frame = ttk.LabelFrame(control_panel, text="Gestion des Particules")
        particle_frame.pack(pady=10, padx=10, fill=tk.X)
        self.create_particle_widgets(particle_frame)

        # --- Section Onglets Simulations ---
        self.notebook = ttk.Notebook(control_panel)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')

        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)

        # --- Panneau de Plot (droite) ---
        plot_panel = ttk.Frame(main_paned_window)
        main_paned_window.add(plot_panel, weight=3) # Donne plus de place au plot

        # --- Zone Matplotlib ---
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Barre d'outils Matplotlib ---
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Barre de Statut ---
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Widgets pour la gestion des particules ---
    def create_particle_widgets(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill=tk.X)

        ttk.Label(input_frame, text="Masse (u):").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.mass_entry = ttk.Entry(input_frame, width=10)
        self.mass_entry.grid(row=0, column=1, padx=5)
        self.mass_entry.insert(0, "1.0")

        ttk.Label(input_frame, text="Charge (e):").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.charge_entry = ttk.Entry(input_frame, width=10)
        self.charge_entry.grid(row=0, column=3, padx=5)
        self.charge_entry.insert(0, "1.0")

        add_btn = ttk.Button(input_frame, text="Ajouter", command=self.add_particle)
        add_btn.grid(row=0, column=4, padx=10)

        # Treeview pour afficher les particules
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        self.particle_tree = ttk.Treeview(tree_frame, columns=('Mass (u)', 'Charge (e)'), show='headings', height=5)
        self.particle_tree.heading('Mass (u)', text='Masse (u)')
        self.particle_tree.heading('Charge (e)', text='Charge (e)')
        self.particle_tree.column('Mass (u)', width=80, anchor=tk.CENTER)
        self.particle_tree.column('Charge (e)', width=80, anchor=tk.CENTER)

        # Scrollbar pour Treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar.set)

        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer sélection", command=self.remove_particle)
        remove_btn.pack(pady=5)

    # --- Widgets pour la déviation magnétique ---
    def create_magnetic_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Vitesse initiale
        self.v0_mag_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(frame, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)

        # Champ Magnétique (Slider)
        ttk.Label(frame, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        slider_frame_bz = ttk.Frame(frame)
        slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=0.1)
        self.bz_slider = ttk.Scale(slider_frame_bz, from_=0, to=1.0, orient=tk.HORIZONTAL, variable=self.bz_var, command=self._update_bz_label)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(slider_frame_bz, textvariable=self.bz_label_var, width=10).pack(side=tk.LEFT)

        # Domaine x
        self.xmin_mag_var = tk.StringVar(value="0.0")
        self.add_labeled_entry(frame, "X min (m):", self.xmin_mag_var).pack(fill=tk.X, pady=3)
        self.xmax_mag_var = tk.StringVar(value="0.5")
        self.add_labeled_entry(frame, "X max (m):", self.xmax_mag_var).pack(fill=tk.X, pady=3)

        # Bouton Tracer
        trace_btn = ttk.Button(frame, text="Tracer Déviation Magnétique", command=self.run_magnetic_simulation)
        trace_btn.pack(pady=15)

    def _update_bz_label(self, event=None):
        self.bz_label_var.set(f"{self.bz_var.get():.3f} T")

    # --- Widgets pour la déviation électrique ---
    def create_electric_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Vitesse initiale
        self.v0_elec_var = tk.StringVar(value="1e5")
        self.add_labeled_entry(frame, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)

        # Angle initial
        self.angle_var = tk.StringVar(value="30.0") # En degrés pour l'utilisateur
        self.add_labeled_entry(frame, "Angle Initial (° vs y):", self.angle_var).pack(fill=tk.X, pady=3)

        # Hauteur initiale
        self.y0_var = tk.StringVar(value="0.15")
        self.add_labeled_entry(frame, "Hauteur Initiale (m):", self.y0_var).pack(fill=tk.X, pady=3)

        # Potentiel (Slider)
        ttk.Label(frame, text="Diff. Potentiel Plaque (V):").pack(anchor=tk.W, pady=(5,0))
        slider_frame_v = ttk.Frame(frame)
        slider_frame_v.pack(fill=tk.X, pady=(0,5))
        self.pot_var = tk.DoubleVar(value=-5000)
        self.pot_slider = ttk.Scale(slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._update_pot_label)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.0f} V")
        ttk.Label(slider_frame_v, textvariable=self.pot_label_var, width=10).pack(side=tk.LEFT)

        # Distance entre plaques
        self.dist_var = tk.StringVar(value="0.15") # Doit correspondre à y0 si la plaque est à y=0
        self.add_labeled_entry(frame, "Distance Plaque (m):", self.dist_var).pack(fill=tk.X, pady=3)

        # Bouton Tracer
        trace_btn = ttk.Button(frame, text="Tracer Déviation Électrique", command=self.run_electric_simulation)
        trace_btn.pack(pady=15)

    def _update_pot_label(self, event=None):
        self.pot_label_var.set(f"{self.pot_var.get():.0f} V")

    # --- Helper pour ajouter Label + Entry ---
    def add_labeled_entry(self, parent, label_text, string_var):
        entry_frame = ttk.Frame(parent)
        ttk.Label(entry_frame, text=label_text, width=20).pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return entry_frame

    # --- Logique métier ---
    def add_particle(self):
        try:
            mass_u = float(self.mass_entry.get())
            charge_e = float(self.charge_entry.get())

            if mass_u <= 0 or charge_e == 0: # Masse doit être > 0, charge != 0
                raise ValueError("Masse > 0 et Charge != 0 requis.")

            particle_info = (mass_u, charge_e)
            if particle_info not in self.particles_data :
                self.particles_data.append(particle_info)
                # Ajouter à Treeview
                self.particle_tree.insert('', tk.END, values=(f"{mass_u:.2f}", f"{charge_e:.2f}"))
            else :
                messagebox.showerror("Erreur d'entrée", "On ne peut pas rajouter 2 fois la même particule")
                self.status_var.set("Erreur d'ajout de particule.")

            

            # Optionnel: Réinitialiser les champs
            # self.mass_entry.delete(0, tk.END)
            # self.charge_entry.delete(0, tk.END)
            self.status_var.set(f"Particule ajoutée: {mass_u:.2f} u, {charge_e:.2f} e")

        except ValueError as e:
            messagebox.showerror("Erreur d'entrée", f"Entrée invalide : {e}")
            self.status_var.set("Erreur d'ajout de particule.")

    def remove_particle(self):
        selected_items = self.particle_tree.selection()
        if not selected_items:
            messagebox.showwarning("Aucune sélection", "Veuillez sélectionner une particule à supprimer.")
            return

        indices_to_remove = []
        items_to_remove_tree = []

        for item_id in selected_items:
            index = self.particle_tree.index(item_id)
            indices_to_remove.append(index)
            items_to_remove_tree.append(item_id)

        # Supprimer de la liste de données (en partant de la fin pour éviter les pbs d'index)
        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del self.particles_data[index]

        # Supprimer du Treeview
        for item_id in items_to_remove_tree:
            self.particle_tree.delete(item_id)

        self.status_var.set(f"{len(selected_items)} particule(s) supprimée(s).")


    def run_magnetic_simulation(self):
        if not self.particles_data:
            messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            return

        try:
            # Récupérer et valider les paramètres
            v0 = float(self.v0_mag_var.get())
            bz = self.bz_var.get() # Directement du slider
            # x_min = float(self.xmin_mag_var.get())
            # x_max = float(self.xmax_mag_var.get())

            # if v0 <= 0 or x_max <= x_min:
            #     raise ValueError("V0 > 0 et X max > X min requis.")

            # Convertir les particules pour le backend (liste de m/q en kg/C)
    
            # Préparer le plot
            self.ax.cla() # Effacer l'axe précédent
            self.status_var.set("Calcul déviation magnétique en cours...")
            self.root.update_idletasks() # Mettre à jour l'UI

            # Appeler la fonction de traçage modifiée
            print(self.particles_data)
            partie_electroaimant.tracer_ensemble_trajectoires(
                self.particles_data, v0, bz, 1e-4, create_plot = False, ax=self.ax
            )

            # Mettre à jour le canvas
            self.ax.relim() # Recalculer les limites si nécessaire
            self.ax.autoscale_view() # Ajuster la vue
            self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")

        except ValueError as e:
            messagebox.showerror("Erreur de paramètre", f"Paramètre invalide : {e}")
            self.status_var.set("Erreur de simulation magnétique.")
        # except Exception as e:
        #     messagebox.showerror("Erreur de Simulation", f"Une erreur est survenue: {e}")
        #     self.status_var.set("Erreur de simulation magnétique.")


    def run_electric_simulation(self):
        if not self.particles_data:
            messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            return

        try:
            # Récupérer et valider les paramètres
            v0 = float(self.v0_elec_var.get())
            angle_deg = float(self.angle_var.get())
            y0 = float(self.y0_var.get())
            potentiel = self.pot_var.get() # Directement du slider
            distance = float(self.dist_var.get())

            if v0 <= 0 or y0 <= 0 or distance <= 0:
                 raise ValueError("V0 > 0, Hauteur > 0 et Distance > 0 requis.")
            if not (0 <= angle_deg < 90): # Angle typiquement aigu par rapport à y
                 raise ValueError("Angle doit être entre 0° et 90°.")

            # Convertir angle en radians
            angle_rad = np.radians(angle_deg)

            # Calculer le champ électrique
            E = potentiel / distance # E est dirigé de +V vers -V

            # Convertir les particules pour le backend (liste de tuples (masse_u, charge_eV))
            # Ici on utilise directement la charge en eV comme demandé par la classe particule de deviation.py
            particles_ueV = [(p['mass_u'], p['charge_e']) for p in self.particles_data]

            # Préparer le plot
            self.ax.cla() # Effacer l'axe précédent
            self.status_var.set("Calcul déviation électrique en cours...")
            self.root.update_idletasks() # Mettre à jour l'UI

            # Appeler la fonction de traçage modifiée
            deviation.tracer_ensemble_trajectoires(
                particles_ueV, v0, angle_rad, y0, E, ax=self.ax
            )

            # Mettre à jour le canvas
            # Les limites sont gérées dans la fonction de traçage modifiée pour ce cas
            self.canvas.draw()
            self.status_var.set("Tracé déviation électrique terminé.")

        except ValueError as e:
            messagebox.showerror("Erreur de paramètre", f"Paramètre invalide : {e}")
            self.status_var.set("Erreur de simulation électrique.")
        except Exception as e:
            messagebox.showerror("Erreur de Simulation", f"Une erreur est survenue: {e}")
            self.status_var.set("Erreur de simulation électrique.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()