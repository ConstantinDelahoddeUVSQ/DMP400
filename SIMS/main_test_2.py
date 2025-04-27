import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{folder}/Partie Bleue (accélération)/Code")
sys.path.append(f"{folder}//Partie Verte (déviation magnétique)/Code")

try:
    import deviation # type: ignore
    import partie_electroaimant # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que les chemins sys.path sont corrects et que les fichiers existent.")
    sys.exit(1)


class ParticleApp:
    def __init__(self, root):
        """
        Définition de la classe ParticleApp

        Parameters
        ----------
        root : tkinter.Tk (ou tk.Frame)	
            Fenêtre principale Tkinter dans laquelle l'application sera construite.
        """
        self.root = root
        self.root.title("Simulateur SIMS - Déviations")
        self.root.geometry("1500x800") # Taille initiale

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
        """
        Widget pour la création des particules

        Parameters
        ----------
        parent : tkinter.frame
            Le conteneur (Frame) dans lequel placer les widgets.
        """
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
        """
        Widgets pour la déviation magnétique

        Parameters
        ----------
        parent : tkinter.frame
            Le conteneur (Frame) dans lequel placer les widgets.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        self.dynamic_inputs_frame = ttk.Frame(frame)
        self.base_inputs_frame = ttk.Frame(frame)

        self.x_detecteur_var = tk.StringVar(value="1e-4")
        self.add_labeled_entry(frame, "X détecteur (m) :", self.x_detecteur_var).pack(fill=tk.X, pady=3)

        # Checkbox "Tracer dynamiquement"
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Tracer dynamiquement", variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs)
        dynamic_check.pack(anchor=tk.W, pady=5)


        # Inside base frame : 

        # Vitesse initiale
        self.v0_mag_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(self.base_inputs_frame, "Vitesse Initiale (m/s) : ", self.v0_mag_var).pack(fill=tk.X, pady=3)

        # Bz
        self.bz_mag_var = tk.StringVar(value="1")
        self.add_labeled_entry(self.base_inputs_frame, "Champ magnétique absolu (T) : ", self.bz_mag_var).pack(fill=tk.X, pady=3)

        # Bouton Tracer
        trace_btn = ttk.Button(self.base_inputs_frame, text="Tracer Déviation Magnétique", command=self.run_magnetic_simulation)
        trace_btn.pack(pady=15)
        

        # Inside dynamic frame : 

        # Bz
        self.bz_min_var = tk.StringVar(value="0.001")
        self.add_labeled_entry(self.dynamic_inputs_frame, "Bz min (T) :", self.bz_min_var).pack(fill=tk.X, pady=3)
        self.bz_max_var = tk.StringVar(value="0.5")
        self.add_labeled_entry(self.dynamic_inputs_frame, "Bz max (T) :", self.bz_max_var).pack(fill=tk.X, pady=3)
        
        # V0
        self.v0_min_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(self.dynamic_inputs_frame, "V0 min (m/s) :", self.v0_min_var).pack(fill=tk.X, pady=3)
        self.v0_max_var = tk.StringVar(value="1e7")
        self.add_labeled_entry(self.dynamic_inputs_frame, "V0 max (m/s) :", self.v0_max_var).pack(fill=tk.X, pady=3)

        # Champ Magnétique (Slider)
        ttk.Label(self.dynamic_inputs_frame, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_bz = ttk.Frame(self.dynamic_inputs_frame)
        self.slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=1)
        self.bz_slider = ttk.Scale(self.slider_frame_bz, from_=0.001, to=1.0, orient=tk.HORIZONTAL, variable=self.bz_var, command=self._on_bz_slider_change)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(self.slider_frame_bz, textvariable=self.bz_label_var, width=15).pack(side=tk.LEFT)

        # Vitesse initiale (Slider)
        ttk.Label(self.dynamic_inputs_frame, text="Vitesse initiale (m/s):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_v0 = ttk.Frame(self.dynamic_inputs_frame)
        self.slider_frame_v0.pack(fill=tk.X, pady=(0,5))
        self.v0_var = tk.DoubleVar(value=1e6)
        self.v0_slider = ttk.Scale(self.slider_frame_v0, from_=1e6, to=1e7, orient=tk.HORIZONTAL, variable=self.v0_var, command=self._on_v0_slider_change)
        self.v0_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var = tk.StringVar(value=f"{self.v0_var.get():.2e} (m/s)")
        ttk.Label(self.slider_frame_v0, textvariable=self.v0_label_var, width=15).pack(side=tk.LEFT)

        self.base_inputs_frame.pack(fill=tk.X, pady=5)
        # Bouton Tracer
        trace_btn = ttk.Button(self.dynamic_inputs_frame, text="Tracer Déviation Magnétique", command=self.run_magnetic_simulation)
        trace_btn.pack(pady=15)

    def toggle_dynamic_inputs(self):
        if self.dynamic_trace_var.get():
            self.base_inputs_frame.pack_forget()
            self.dynamic_inputs_frame.pack(fill=tk.X, pady=5)
        else:
            self.dynamic_inputs_frame.pack_forget()
            self.base_inputs_frame.pack(fill=tk.X, pady=5)

        # --- NOUVELLE FONCTION CALLBACK pour slider Bz ---
    def _on_bz_slider_change(self, event=None):
        """
        Callback pour slider Bz
        Appelé lorsque le slider Bz est modifié.
        
        Parameters
        ----------
        event : Event ou None
        """
        self._update_bz_label() # Met à jour le label texte
        # Lance la simulation seulement s'il y a des particules
        if self.particles_data:
            # Pas besoin de update_idletasks ici, run_magnetic_simulation le fera
            self.run_magnetic_simulation(called_by_slider=True) # Indique d'où vient l'appel

    def _update_bz_label(self, event=None):
        """
        Label du champ Bz

        Parameters
        ----------
        event : Event ou None
        """

        self.bz_label_var.set(f"{self.bz_var.get():.3f} T")

    def _on_v0_slider_change(self, event=None):
        """
        Slider de v0, la vitesse initiale
        Appelé lorsque le slider v0 est modifié.

        Parameters
        ----------
        event : Event ou None
        """

        self._update_v0_label() # Met à jour le label texte
        # Lance la simulation seulement s'il y a des particules
        if self.particles_data:
            # Pas besoin de update_idletasks ici, run_magnetic_simulation le fera
            self.run_magnetic_simulation(called_by_slider=True) # Indique d'où vient l'appel

    def _update_v0_label(self, event=None):
        """
        Label de v0

        Parameters
        ----------
        event : Event ou None
        """
        self.v0_label_var.set(f"{self.v0_var.get():.2e} (m/s)")

    # --- Widgets pour la déviation électrique ---
    def create_electric_widgets(self, parent):
        """
        Widgets pour la déviation électrique

        Parameters
        ----------
        parent : tkinter.frame
            Le conteneur (Frame) dans lequel placer les widgets.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Vitesse initiale
        self.v0_elec_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(frame, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)

        # Angle initial
        self.angle_var = tk.StringVar(value="30.0") # En degrés pour l'utilisateur
        self.add_labeled_entry(frame, "Angle Initial (° vs y):", self.angle_var).pack(fill=tk.X, pady=3)

        # Hauteur initiale
        self.y0_var = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Hauteur Initiale (m):", self.y0_var).pack(fill=tk.X, pady=3)

        # Potentiel (Slider)
        ttk.Label(frame, text="Diff. Potentiel Plaque (V):").pack(anchor=tk.W, pady=(5,0))
        slider_frame_v = ttk.Frame(frame)
        slider_frame_v.pack(fill=tk.X, pady=(0,5))
        self.pot_var = tk.DoubleVar(value=-5000)

        self.pot_slider = ttk.Scale(slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.0f} V")
        ttk.Label(slider_frame_v, textvariable=self.pot_label_var, width=10).pack(side=tk.LEFT)

        # Distance entre plaques
        self.dist_var = tk.StringVar(value="0.05") 
        self.add_labeled_entry(frame, "Distance Plaque (m):", self.dist_var).pack(fill=tk.X, pady=3)

        # Bouton Tracer (reste utile pour lancer après modif de V0, angle, y0, dist)
        trace_btn = ttk.Button(frame, text="Tracer Déviation Électrique", command=self.run_electric_simulation)
        trace_btn.pack(pady=15)

    # --- NOUVELLE FONCTION CALLBACK pour slider Potentiel ---
    def _on_pot_slider_change(self, event=None):
        """
        Callback pour slider le potentiel
        Appelé lorsque le slider potentiel est modifié.
        
        Parameters
        ----------
        event : Event ou None
        """
        self._update_pot_label() # Met à jour le label texte
        # Lance la simulation seulement s'il y a des particules
        if self.particles_data:
             # Pas besoin de update_idletasks ici, run_electric_simulation le fera
            self.run_electric_simulation(called_by_slider=True) # Indique d'où vient l'appel

    def _update_pot_label(self, event=None):
        """
        Met à jour le label du slider Potentiel.
        
        Parameters
        ----------
        event : Event ou None
        """
        self.pot_label_var.set(f"{self.pot_var.get():.0f} V")

    # --- Helper pour ajouter Label + Entry ---
    def add_labeled_entry(self, parent, label_text, string_var):
        """
        Helper pour ajouter Label + Entry.
        
        Parameters
        ----------
        parent : tkinter.frame
            Le conteneur (Frame) dans lequel placer les widgets.
        label_text : string
            texte de la légende
        string_var : string
        """
        entry_frame = ttk.Frame(parent)
        ttk.Label(entry_frame, text=label_text, width=20).pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return entry_frame

    # --- Logique métier ---
    def add_particle(self):
        """
        Logique métier

        Parameters
        ----------
        """
        try:
            mass_u = float(self.mass_entry.get())
            charge_e = float(self.charge_entry.get())

            if mass_u <= 0:
                raise ValueError("Masse doit être > 0.")
            if len(self.particles_data) > 0 :
                if charge_e * self.particles_data[0][1] <= 0 :
                    raise ValueError("Veuillez rentrer des particules de charge identique")

            particle_info = (mass_u, charge_e)
            if particle_info not in self.particles_data :
                self.particles_data.append(particle_info)
                self.particle_tree.insert('', tk.END, values=(f"{mass_u:.3f}", f"{charge_e:+.2f}")) # Formatage amélioré
                self.status_var.set(f"Particule ajoutée: {mass_u:.3f} u, {charge_e:+.2f} e")
            else :
                messagebox.showwarning("Doublon", "Cette particule est déjà dans la liste.")
                self.status_var.set("Ajout annulé (doublon).")

        except ValueError as e:
            messagebox.showerror("Erreur d'entrée", f"Entrée invalide : {e}")
            self.status_var.set("Erreur d'ajout de particule.")

    def remove_particle(self):
        """
        Enlever la particule.
        
        Parameters
        ----------
        """
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


    def run_magnetic_simulation(self, called_by_slider=False):
        """
        Simulation du champ magnétique ()
        
        Parameters
        ----------
        called_by_slider : booléen
        """
        if not self.particles_data:
            if not called_by_slider: # N'affiche le message que si le bouton est cliqué
                messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla() # Efface le graphe s'il n'y a pas de particules
            self.canvas.draw()
            return

        try:
            allow_trace = True
            # Récupérer et valider les paramètres
            x_detecteur = float(self.x_detecteur_var.get())
            
            if not self.dynamic_trace_var.get() :
                v0 = float(self.v0_mag_var.get())
                bz = float(self.bz_mag_var.get())
                if v0 <= 0 :
                    messagebox.showerror("Valeur incorrecte", "La valeur de vitesse initiale ne peut être négative ou nulle")
                    self.status_var.set("Erreur de simulation magnétique.")
                    allow_trace = False
                elif bz <= 0 :
                    messagebox.showerror("Valeur incorrecte", "La valeur de champ magnétique ne peut être négative ou nulle")
                    self.status_var.set("Erreur de simulation magnétique.")
                    allow_trace = False
            else :
                if not called_by_slider : 
                    bz_min = float(self.bz_min_var.get())
                    bz_max = float(self.bz_max_var.get())
                    v0_min = float(self.v0_min_var.get())
                    v0_max = float(self.v0_max_var.get())
                    if v0_min <= 0 :
                        messagebox.showerror("Valeur incorrecte", "La valeur de vitesse initiale minimale ne peut être négative ou nulle")
                        self.status_var.set("Erreur de simulation magnétique.")
                        allow_trace = False
                    elif bz_min <= 0 :
                        messagebox.showerror("Valeur incorrecte", "La valeur de champ magnétique minimale ne peut être négative ou nulle")
                        self.status_var.set("Erreur de simulation magnétique.")
                        allow_trace = False
                    elif v0_min >= v0_max :
                        messagebox.showerror("Valeur incorrecte", "La valeur de vitesse initiale maximale ne peut inférieure ou égale à la valeur minimale")
                        self.status_var.set("Erreur de simulation magnétique.")
                        allow_trace = False
                    elif bz_min >= bz_max :
                        messagebox.showerror("Valeur incorrecte", "La valeur de champ magnétique maximale ne peut inférieure ou égale à la valeur minimale")
                        self.status_var.set("Erreur de simulation magnétique.")
                        allow_trace = False
                    if allow_trace : 
                        self.bz_slider.config(from_=bz_min, to=bz_max)
                        self.bz_var.set((bz_max - bz_min) / 2)
                        self.bz_label_var.set(f"{self.bz_var.get():.3f} T")
                        self.v0_slider.config(from_=v0_min, to=v0_max)
                        self.v0_var.set((v0_max - v0_min) / 2)
                        self.v0_label_var.set(f"{self.v0_var.get():.2e} (m/s)")

            if allow_trace : 
                # Préparer le plot
                self.ax.cla() # Effacer l'axe précédent
                self.status_var.set("Calcul déviation magnétique en cours...")
                self.root.update_idletasks() # Mettre à jour l'UI
                
                if self.dynamic_trace_var.get() : 
                    v0, bz = self.v0_var.get(), self.bz_var.get()
                partie_electroaimant.tracer_ensemble_trajectoires(
                        self.particles_data, v0, bz, x_detecteur, create_plot = False, ax=self.ax
                    )

                # Mettre à jour le canvas
                self.ax.relim() # Recalculer les limites si nécessaire
                self.ax.autoscale_view() # Ajuster la vue
                self.canvas.draw()
                self.status_var.set("Tracé déviation magnétique terminé.")

        except ValueError as e:
            messagebox.showerror("Erreur de paramètre", f"Paramètre invalide : {e}")
            self.status_var.set("Erreur de simulation magnétique.")
        except Exception as e:
            messagebox.showerror("Erreur de Simulation", f"Une erreur est survenue: {e}")
            self.status_var.set("Erreur de simulation magnétique.")


    def run_electric_simulation(self, called_by_slider=False):
        """
        Simulation du champ électrique ()
        
        Parameters
        ----------
        called_by_slider : booléen
        """
        if not self.particles_data:
            if not called_by_slider:
                messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla() # Efface le graphe
            self.canvas.draw()
            return

        # try:
        if True :
            # Récupérer et valider les paramètres
            v0 = float(self.v0_elec_var.get())
            angle_deg = float(self.angle_var.get())
            y0 = float(self.y0_var.get())
            potentiel = self.pot_var.get()
            distance = float(self.dist_var.get())

            if v0 <= 0 : raise ValueError("V0 > 0 requis.")
            if y0 <= 0 : raise ValueError("Hauteur Initiale > 0 requis.")
            if distance <= 0 : raise ValueError("Distance Plaque > 0 requis.")
            if not (0 < angle_deg < 90):
                raise ValueError("Angle doit être entre 0° et 90°.")

            # Convertir angle en radians
            angle_rad = np.radians(angle_deg)

            # Calculer le champ électrique signé
            E = deviation.champ_electrique_v2(distance, potentiel) # Utilise la fonction du module

            # Préparer le plot
            self.ax.cla() # Effacer l'axe précédent
            self.status_var.set("Calcul déviation électrique en cours...")
            self.root.update_idletasks() # Mettre à jour l'UI avant calcul

            # Appeler la fonction de traçage
            deviation.tracer_ensemble_trajectoires(
                self.particles_data, v0, potentiel, angle_rad, distance, create_plot = False, ax=self.ax
            )

            # Mettre à jour le canvas
            # Les limites sont gérées dans la fonction de traçage modifiée
            self.canvas.draw()
            self.status_var.set("Tracé déviation électrique terminé.")

        # except ValueError as e:
        #     if not called_by_slider:
        #         messagebox.showerror("Erreur de paramètre", f"Paramètre invalide : {e}")
        #     self.status_var.set(f"Erreur paramètre (Elec): {e}")
        # except Exception as e:
        #     if not called_by_slider:
        #         messagebox.showerror("Erreur de Simulation", f"Une erreur est survenue (Elec): {e}")
        #     print(f"Erreur Simulation Électrique: {e}") # Toujours logger l'erreur
        #     self.status_var.set("Erreur de simulation électrique.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()