import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(folder, "Partie Bleue (accélération)", "Code"))
sys.path.append(os.path.join(folder, "Partie Verte (déviation magnétique)", "Code"))

try:
    import deviation # type: ignore
    import partie_electroaimant # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print(f"Vérifiez l'existence des fichiers dans:")
    print(f"  {os.path.join(folder, 'Partie Bleue (accélération)', 'Code')}")
    print(f"  {os.path.join(folder, 'Partie Verte (déviation magnétique)', 'Code')}")
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

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabelframe.Label", font=('Helvetica', 12, 'bold'))
        style.configure("TLabel", padding=2)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))

        # Liste de toutes les particules (masse, charge)
        self.particles_data = []

        # Structure principale
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # Panneau de contrôle
        control_panel = ttk.Frame(main_paned_window, width=400)
        main_paned_window.add(control_panel, weight=1)

        # Section Particules
        particle_frame = ttk.LabelFrame(control_panel, text="Gestion des Particules")
        particle_frame.pack(pady=10, padx=10, fill=tk.X)
        self.create_particle_widgets(particle_frame)

        # Section Onglets Simulations
        self.notebook = ttk.Notebook(control_panel)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')

        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)

        # Panneau Plot
        plot_panel = ttk.Frame(main_paned_window)
        main_paned_window.add(plot_panel, weight=3) # Donne plus de place au plot

        # Zone Matplotlib
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Barre d'outils Matplotlib
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Barre de Statut
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_closing(self):
        """
        Fonction appelée lorsque l'utilisateur ferme la fenêtre.
        Nettoie les ressources Matplotlib et Tkinter.
        """
        try:
            # Ferme la figure Matplotlib pour libérer ses ressources
            plt.close(self.fig)
        except Exception as e:
            print(f"Erreur lors de la fermeture de la figure Matplotlib: {e}")

        try:
            # Détruit la fenêtre principale Tkinter et termine proprement
            self.root.destroy()
        except Exception as e:
            print(f"Erreur lors de la destruction de la fenêtre Tkinter: {e}")

    # Widgets pour la gestion des particules
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
        
        ttk.Label(input_frame, text="Quelques molécules caractéristiques :").grid(row=0, column=0, padx=5, sticky=tk.W)
        btns_frame = ttk.Frame(input_frame)
        btns_frame.pack(side=tk.LEFT, padx=10)

        btn_o2 = ttk.Button(btns_frame, text="O₂", command=lambda: self.ajt_particle_connue(32.0, -2.0))
        btn_o2.pack(side=tk.LEFT, padx=5)

        btn_si = ttk.Button(btns_frame, text="Si", command=lambda: self.ajt_particle_connue(28.0, +1.0))
        btn_si.pack(side=tk.LEFT, padx=5)

        create_molecule_btn = ttk.Button(parent, text="Créer ma particule chargée", command=self.ouvrir_fenetre_tp)
        create_molecule_btn.pack(pady=5)


        # Bloc pour afficher les particules
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        self.particle_tree = ttk.Treeview(tree_frame, columns=('Mass (u)', 'Charge (e)'), show='headings', height=5)
        self.particle_tree.heading('Mass (u)', text='Masse (u)')
        self.particle_tree.heading('Charge (e)', text='Charge (e)')
        self.particle_tree.column('Mass (u)', width=80, anchor=tk.CENTER)
        self.particle_tree.column('Charge (e)', width=80, anchor=tk.CENTER)

        # Scrollbar pour le bloc
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar.set)

        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer sélection", command=self.remove_particle)
        remove_btn.pack(pady=5)

    def ouvrir_fenetre_tp(self):
            """
            Ouvre une nouvelle fenêtre pour créer une molécule à partir du tableau périodique.
            """
            self.molecule_fenetre = tk.Toplevel(self.root)
            self.molecule_fenetre.title("Construire une molécule")
            self.molecule_fenetre.geometry("1000x600")


            self.selected_elts = {}  

            periodic_layout = [
    [('H', 1.008)] + [None]*16 + [('He', 4.0026)],
    [('Li', 6.94), ('Be', 9.0122)] + [None]*10 + [('B', 10.81), ('C', 12.011), ('N', 14.007), ('O', 15.999), ('F', 18.998), ('Ne', 20.180)],
    [('Na', 22.990), ('Mg', 24.305)] + [None]*10 + [('Al', 26.982), ('Si', 28.085), ('P', 30.974), ('S', 32.06), ('Cl', 35.45), ('Ar', 39.948)],
    [('K', 39.098), ('Ca', 40.078), ('Sc', 44.956), ('Ti', 47.867), ('V', 50.942), ('Cr', 51.996),
     ('Mn', 54.938), ('Fe', 55.845), ('Co', 58.933), ('Ni', 58.693), ('Cu', 63.546), ('Zn', 65.38),
     ('Ga', 69.723), ('Ge', 72.630), ('As', 74.922), ('Se', 78.971), ('Br', 79.904), ('Kr', 83.798)],
    [('Rb', 85.468), ('Sr', 87.62), ('Y', 88.906), ('Zr', 91.224), ('Nb', 92.906), ('Mo', 95.95),
     ('Tc', 98.0), ('Ru', 101.07), ('Rh', 102.91), ('Pd', 106.42), ('Ag', 107.87), ('Cd', 112.41),
     ('In', 114.82), ('Sn', 118.71), ('Sb', 121.76), ('Te', 127.60), ('I', 126.90), ('Xe', 131.29)],
    [('Cs', 132.91), ('Ba', 137.33)] + [None]*2 + 
    [('Hf', 178.49), ('Ta', 180.95), ('W', 183.84), ('Re', 186.21), ('Os', 190.23), ('Ir', 192.22),
     ('Pt', 195.08), ('Au', 196.97), ('Hg', 200.59), ('Tl', 204.38), ('Pb', 207.2), ('Bi', 208.98),
     ('Po', 209.0), ('At', 210.0), ('Rn', 222.0)],
    [('Fr', 223.0), ('Ra', 226.0)] + [None]*2 +
    [('Rf', 267.0), ('Db', 270.0), ('Sg', 271.0), ('Bh', 270.0), ('Hs', 277.0), ('Mt', 278.0),
     ('Ds', 281.0), ('Rg', 282.0), ('Cn', 285.0), ('Nh', 286.0), ('Fl', 289.0), ('Mc', 290.0),
     ('Lv', 293.0), ('Ts', 294.0), ('Og', 294.0)],
    [None]*2 + [('La', 138.91), ('Ce', 140.12), ('Pr', 140.91), ('Nd', 144.24), ('Pm', 145.0),
     ('Sm', 150.36), ('Eu', 151.96), ('Gd', 157.25), ('Tb', 158.93), ('Dy', 162.50), ('Ho', 164.93),
     ('Er', 167.26), ('Tm', 168.93), ('Yb', 173.05), ('Lu', 174.97)],
    [None]*2 + [('Ac', 227.0), ('Th', 232.04), ('Pa', 231.04), ('U', 238.03), ('Np', 237.0),
     ('Pu', 244.0), ('Am', 243.0), ('Cm', 247.0), ('Bk', 247.0), ('Cf', 251.0), ('Es', 252.0),
     ('Fm', 257.0), ('Md', 258.0), ('No', 259.0), ('Lr', 262.0)],
]

            

            table_frame = ttk.Frame(self.molecule_fenetre)
            table_frame.pack(pady=10, padx=10)

            for row_idx, row in enumerate(periodic_layout):
                for col_idx, element in enumerate(row):
                    if element:
                        symbol, mass = element
                        btn = ttk.Button(table_frame, text=symbol, width=4,
                                        command=lambda s=symbol, m=mass: self.construction_de_molecule(s, m))
                        btn.grid(row=row_idx, column=col_idx, padx=2, pady=2)


            self.molecule_display_var = tk.StringVar(value="Molécule : (vide)")
            ttk.Label(self.molecule_fenetre, textvariable=self.molecule_display_var).pack(pady=10)

            charge_frame = ttk.Frame(self.molecule_fenetre)
            charge_frame.pack(pady=5)

            ttk.Label(charge_frame, text="Charge (e) :").pack(side=tk.LEFT)
            self.molecule_charge_var = tk.StringVar(value="1.0")
            charge_entry = ttk.Entry(charge_frame, textvariable=self.molecule_charge_var, width=10)
            charge_entry.pack(side=tk.LEFT, padx=5)

            submit_btn = ttk.Button(self.molecule_fenetre, text="Soumettre molécule", command=self.submit_molecule)
            submit_btn.pack(pady=10)


    def construction_de_molecule(self, symbol, mass):
            """
            Ajoute un élément à la molécule en cours.
            """
            if symbol in self.selected_elts:
                self.selected_elts[symbol]['count'] += 1
            else:
                self.selected_elts[symbol] = {'mass': mass, 'count': 1}

            molecule_text = " + ".join(f"{v['count']}{k}" for k, v in self.selected_elts.items())
            self.molecule_display_var.set(f"Molécule : {molecule_text}")

    def ajt_particle_connue(self, mass_u, charge_e):
            """
            Ajoute une particule prédéfinie (ex: O₂, Si) directement dans la liste.

            Parameters
            ----------
            mass_u : float
                Masse en unité atomique
            charge_e : float
                Charge en multiple de e
            """
            try:
                if mass_u <= 0:
                    raise ValueError("Masse doit être > 0.")
                if len(self.particles_data) > 0:
                    if charge_e * self.particles_data[0][1] <= 0:
                        raise ValueError("Veuillez rentrer des particules de charge identique")

                particle_info = (mass_u, charge_e)
                if particle_info not in self.particles_data:
                    self.particles_data.append(particle_info)
                    self.particle_tree.insert('', tk.END, values=(f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                    self.status_var.set(f"Particule ajoutée: {mass_u:.3f} u, {charge_e:+.2f} e")
                
            except ValueError as e:
                messagebox.showerror("Erreur d'entrée", f"Entrée invalide : {e}")
                self.status_var.set("Erreur d'ajout de particule.")

    def submit_molecule(self):
        """
        Calcule la masse totale et ajoute la molécule comme particule avec la charge entrée.
        """
        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())

            charge = float(self.molecule_charge_var.get())

            if total_mass <= 0:
                raise ValueError("La masse totale est invalide.")
            if len(self.particles_data) > 0:
                if charge * self.particles_data[0][1] <= 0:
                    raise ValueError("Veuillez rentrer des particules de charge identique.")

            particle_info = (total_mass, charge)
            if particle_info not in self.particles_data:
                self.particles_data.append(particle_info)
                self.particle_tree.insert('', tk.END, values=(f"{total_mass:.3f}", f"{charge:+.2f}"))
                self.status_var.set(f"Molécule ajoutée : {total_mass:.3f} u, {charge:+.2f} e")

            self.molecule_fenetre.destroy()

        except ValueError as e:
            messagebox.showerror("Erreur de saisie", f"Erreur de soumission : {e}")
            self.status_var.set("Erreur lors de la soumission de molécule.")

    # Widgets magnétiques
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

        # Frames pour afficher/cacher
        self.dynamic_inputs_frame = ttk.Frame(frame)
        self.base_inputs_frame = ttk.Frame(frame)

        self.x_detecteur_var = tk.StringVar(value="0.1")
        self.add_labeled_entry(frame, "X détecteur (m) :", self.x_detecteur_var).pack(fill=tk.X, pady=3)

        # Checkbox "Tracer dynamiquement"
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Tracer dynamiquement", variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs)
        dynamic_check.pack(anchor=tk.W, pady=5)

        # --- Widgets Non-Dynamiques ---
        parent_base = self.base_inputs_frame
        self.v0_mag_var = tk.StringVar(value="1e6") # Valeur réaliste
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)
        self.bz_mag_var = tk.StringVar(value="0.2") # Valeur réaliste
        self.add_labeled_entry(parent_base, "Champ magnétique (T):", self.bz_mag_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Déviation Magnétique", command=self.run_magnetic_simulation)
        trace_btn_base.pack(pady=15)

        # --- Widgets Dynamiques ---
        parent_dyn = self.dynamic_inputs_frame
        # Limites Bz
        self.bz_min_var = tk.StringVar(value="0.01")
        self.add_labeled_entry(parent_dyn, "Bz min (T):", self.bz_min_var).pack(fill=tk.X, pady=3)
        self.bz_max_var = tk.StringVar(value="0.5")
        self.add_labeled_entry(parent_dyn, "Bz max (T):", self.bz_max_var).pack(fill=tk.X, pady=3)

        # Slider Bz
        ttk.Label(parent_dyn, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_bz = ttk.Frame(parent_dyn)
        self.slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=0.2) # Init au milieu de la plage par défaut
        # Limites initiales du slider (seront reconfigurées)
        self.bz_slider = ttk.Scale(self.slider_frame_bz, from_=0.01, to=0.5, orient=tk.HORIZONTAL, variable=self.bz_var, command=self._on_bz_slider_change)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(self.slider_frame_bz, textvariable=self.bz_label_var, width=10).pack(side=tk.LEFT) # Largeur réduite

        # Limites v0
        self.v0_min_var = tk.StringVar(value="1e5")
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.v0_min_var).pack(fill=tk.X, pady=3)
        self.v0_max_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.v0_max_var).pack(fill=tk.X, pady=3)

        # Slider V0
        ttk.Label(parent_dyn, text="Vitesse initiale (m/s):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_v0 = ttk.Frame(parent_dyn)
        self.slider_frame_v0.pack(fill=tk.X, pady=(0,5))
        self.v0_var = tk.DoubleVar(value=1e6) # Init au milieu
        # Limites initiales du slider (seront reconfigurées)
        self.v0_slider = ttk.Scale(self.slider_frame_v0, from_=1e5, to=1e7, orient=tk.HORIZONTAL, variable=self.v0_var, command=self._on_v0_slider_change)
        self.v0_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var = tk.StringVar(value=f"{self.v0_var.get():.2e} m/s") # Formatage m/s
        ttk.Label(self.slider_frame_v0, textvariable=self.v0_label_var, width=10).pack(side=tk.LEFT)

        # Bouton Tracer Dynamique
        trace_btn_dyn = ttk.Button(parent_dyn, text="Tracer Déviation Électrique", command=self.run_magnetic_simulation)
        trace_btn_dyn.pack(pady=15)

        # Affichage initial
        self.toggle_dynamic_inputs() # Appelle pour afficher le bon frame au départ


    def toggle_dynamic_inputs(self) :
        """ 
        Fonction pour changer l'affichage du menu lorsque le checkbox dynamique est activé.
        """
        if self.dynamic_trace_var.get():
            self.base_inputs_frame.pack_forget()
            self.dynamic_inputs_frame.pack(fill=tk.X, pady=5, padx=5) # Ajout padx
        else:
            self.dynamic_inputs_frame.pack_forget()
            self.base_inputs_frame.pack(fill=tk.X, pady=5, padx=5) # Ajout padx

    # Callbacks sliders magnétiques (inchangés)
    def _on_bz_slider_change(self, event=None):
        """
        Callback pour slider Bz
        Appelé lorsque le slider Bz est modifié afin de rendre le plot dynamique.
        
        Parameters
        ----------
        event : Event ou None
        """
        self._update_bz_label()
        if self.particles_data:
            self.run_magnetic_simulation(called_by_slider=True)
    def _update_bz_label(self, event=None):
        self.bz_label_var.set(f"{self.bz_var.get():.3f} T")
    def _on_v0_slider_change(self, event=None):
        self._update_v0_label()
        if self.particles_data:
            self.run_magnetic_simulation(called_by_slider=True)
    def _update_v0_label(self, event=None):
        self.v0_label_var.set(f"{self.v0_var.get():.2e} m/s") # Formatage m/s

    # --- Widgets Électriques (CORRIGÉ) ---
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

        # Frames pour afficher/cacher
        self.dynamic_electric_inputs_frame = ttk.Frame(frame)
        self.base_electric_inputs_frame = ttk.Frame(frame)

        # --- Widgets Communs ---
        self.angle_var = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3) # Précisé vs +y
        self.dist_var = tk.StringVar(value="0.1") # Distance/hauteur réaliste
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3) # Nom générique

        # Checkbox commun
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        dynamic_elec_check = ttk.Checkbutton(frame, text="Tracer dynamiquement (potentiel)", variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric)
        dynamic_elec_check.pack(anchor=tk.W, pady=5)

        # --- Widgets Non-Dynamiques (Base) ---
        parent_base = self.base_electric_inputs_frame
        # Potentiel statique
        self.v0_elec_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)

        self.diff_pot_var = tk.StringVar(value="0") # Valeur réaliste
        self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        # Bouton Tracer statique
        trace_btn_base = ttk.Button(parent_base, text="Tracer Déviation Électrique", command=self.run_electric_simulation)
        trace_btn_base.pack(pady=15)

        # --- Widgets Dynamiques ---
        parent_dyn = self.dynamic_electric_inputs_frame

        # Entrées pour les limites de vitesse initale
        self.elec_v0_min_var = tk.StringVar(value="1e4") # Limite réaliste
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.elec_v0_min_var).pack(fill=tk.X, pady=3)
        self.elec_v0_max_var = tk.StringVar(value="1e5") # Limite réaliste
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.elec_v0_max_var).pack(fill=tk.X, pady=3)

        # Slider Vitesse initiale
        ttk.Label(parent_dyn, text="Vitesse initiale V0 (m/s):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v0_elec = ttk.Frame(parent_dyn)
        self.slider_frame_v0_elec.pack(fill=tk.X, pady=(0, 5))
        self.v0_var_elec = tk.DoubleVar(value=(float(self.elec_v0_max_var.get()) - float(self.elec_v0_min_var.get())) / 2) # Init au milieu
        # Limites initiales (seront reconfigurées)
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=self.elec_v0_min_var.get(), to=self.elec_v0_max_var.get(), orient=tk.HORIZONTAL, variable=self.v0_var_elec, command=self._on_v0_slider_change_elec)
        self.v0_slider_elec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var_elec = tk.StringVar(value=f"{self.v0_var_elec.get():.2e} (m/s)")
        ttk.Label(self.slider_frame_v0_elec, textvariable=self.v0_label_var_elec, width=15).pack(side=tk.LEFT)


        # Entrées pour les limites du potentiel
        self.diff_pot_min_var = tk.StringVar(value="-5000") # Limite réaliste
        self.add_labeled_entry(parent_dyn, "Potentiel min (V):", self.diff_pot_min_var).pack(fill=tk.X, pady=3)
        self.diff_pot_max_var = tk.StringVar(value="5000") # Limite réaliste
        self.add_labeled_entry(parent_dyn, "Potentiel max (V):", self.diff_pot_max_var).pack(fill=tk.X, pady=3)

        # Slider Potentiel
        ttk.Label(parent_dyn, text="Diff. Potentiel (V):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v = ttk.Frame(parent_dyn)
        self.slider_frame_v.pack(fill=tk.X, pady=(0, 5))
        self.pot_var = tk.DoubleVar(value=-5000) # Init au milieu
        # Limites initiales (seront reconfigurées)
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V")
        ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=15).pack(side=tk.LEFT)

        # Bouton Tracer dynamique
        trace_btn_dyn = ttk.Button(parent_dyn, text="Tracer Déviation Électrique", command=self.run_electric_simulation)
        trace_btn_dyn.pack(pady=15)

        # Affichage initial
        self.toggle_dynamic_electric() # Affiche le bon frame au départ

    def toggle_dynamic_electric(self):
        """ 
        Fonction pour changer l'affichage du menu lorsque la checkbox dynamique du champ électrique est activée.
        """
        if self.dynamic_elec_var.get():
            self.base_electric_inputs_frame.pack_forget()
            self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5) # Ajout padx
        else:
            self.dynamic_electric_inputs_frame.pack_forget()
            self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5) # Ajout padx


    # Callbacks slider potentiel (inchangés)
    def _on_pot_slider_change(self, event=None):
        """
        Callback pour slider le potentiel
        Appelé lorsque le slider potentiel est modifié afin de rendre le plot interactif.
        
        Parameters
        ----------
        event : Event ou None
        """
        self._update_pot_label()
        if self.particles_data:
            self.run_electric_simulation(called_by_slider=True)


    def _on_v0_slider_change_elec(self, event=None) :
        """
        Callback pour slider le v0 partie elec
        Appelé lorsque le slider v0 est modifié afin de rendre le plot interactif.
        
        Parameters
        ----------
        event : Event ou None
        """
        self._update_v0_label_elec()
        if self.particles_data:
            self.run_electric_simulation(called_by_slider=True)

    def _update_pot_label(self, event=None):
        """
        Met à jour le label du slider Potentiel.
        
        Parameters
        ----------
        event : Event ou None
        """
        self.pot_label_var.set(f"{self.pot_var.get():.1f} V")

    def _update_v0_label_elec(self, event=None):
        """
        Met à jour le label du slider v0 (elec).
        
        Parameters
        ----------
        event : Event ou None
        """
        self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e} (m/s)")

    # Helper (inchangé)
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

    # Gestion particules (inchangé, sauf correction signe charge)
    def add_particle(self):
        """
        Fonction appelée lorsqu'une particule est rajoutée pour gérer l'affichage et modifier les variables
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
        Fonction appelée pour enlever une particule.
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

        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del self.particles_data[index]

        # Supprimer du Treeview
        for item_id in items_to_remove_tree:
            self.particle_tree.delete(item_id)

        self.status_var.set(f"{len(selected_items)} particule(s) supprimée(s).")


    # Simulation magnétique
    def run_magnetic_simulation(self, called_by_slider=False):
        """
        Simulation du champ magnétique ()
        
        Parameters
        ----------
        called_by_slider : booléen
        """
        if not self.particles_data:
            if not called_by_slider: 
                messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla() 
            self.canvas.draw()
            return

        try:
            allow_trace = True
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
                self.ax.cla() 
                self.status_var.set("Calcul déviation magnétique en cours...")
                self.root.update_idletasks() 
                
                if self.dynamic_trace_var.get() : 
                    v0, bz = self.v0_var.get(), self.bz_var.get()
                partie_electroaimant.tracer_ensemble_trajectoires(
                        self.particles_data, v0, bz, x_detecteur, create_plot = False, ax=self.ax
                    )

                self.ax.relim() 
                self.ax.autoscale_view() 
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
            if not called_by_slider: messagebox.showwarning("Aucune particule", "Veuillez ajouter au moins une particule.")
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla()
            self.canvas.draw()
            return

        try:
        # if True : 
            angle_deg = float(self.angle_var.get())
            distance = float(self.dist_var.get())

            # Lire le potentiel selon le mode
            if not self.dynamic_elec_var.get(): # Mode Non-Dynamique
                v0 = float(self.v0_elec_var.get())
                potentiel = float(self.diff_pot_var.get())

            else: # Mode Dynamique
                # Si appelé par le slider, lire directement sa valeur
                if called_by_slider:
                    potentiel = self.pot_var.get()
                    v0 = self.v0_var_elec.get()
                # Si appelé par le bouton "Appliquer Limites & Tracer"
                else:
                    pot_min = float(self.diff_pot_min_var.get())
                    pot_max = float(self.diff_pot_max_var.get())
                    v0_min = float(self.elec_v0_min_var.get())
                    v0_max = float(self.elec_v0_max_var.get())

                    # Validation des limites
                    if pot_min >= pot_max : # Permettre min=max si on veut juste fixer une valeur via le mode dyn.
                         raise ValueError("Potentiel max > Potentiel min requis.")
                    # Validation des limites
                    if v0_min >= v0_max : # Permettre min=max si on veut juste fixer une valeur via le mode dyn.
                        raise ValueError("Vitesse initiale max > Vitesse intiale min requis.")
                    elif v0_min <= 0 :
                        raise ValueError("Vitesse initiale min > 0 requis.")

                    # Configurer le slider potentiel
                    self.pot_slider.config(from_=pot_min, to=pot_max)
                    pot_init = (pot_max + pot_min) / 2
                    self.pot_var.set(pot_init)
                    self._update_pot_label() # Met à jour l'affichage

                    # Configurer le slider v0
                    self.v0_slider_elec.config(from_=v0_min, to=v0_max)
                    # Placer la valeur initiale au milieu
                    v0_init = (v0_max + v0_min) / 2
                    self.v0_var_elec.set(v0_init)
                    self._update_v0_label_elec() # Met à jour l'affichage

                    # Lire la valeur initiale du slider pour ce premier tracé
                    potentiel = pot_init
                    v0 = v0_init

            if v0 <= 0 : raise ValueError("V0 > 0 requis.")
            if distance <= 0 : raise ValueError("Hauteur/Distance > 0 requis.")
            if not (0 < angle_deg < 90):
                raise ValueError("Angle doit être entre 0° et 90°.")

            angle_rad = np.radians(angle_deg)

            self.ax.cla()
            self.status_var.set("Calcul déviation électrique en cours...")
            self.root.update_idletasks()

            deviation.tracer_ensemble_trajectoires(self.particles_data, v0, potentiel, angle_rad, distance, create_plot=False, ax=self.ax)

            self.canvas.draw()
            self.status_var.set("Tracé déviation électrique terminé.")

        except ValueError as e: # Attrape aussi l'erreur de conversion float()
            if not called_by_slider: messagebox.showerror("Erreur de paramètre", f"Paramètre invalide (Elec): {e}")
            self.status_var.set(f"Erreur paramètre (Elec): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur de Simulation", f"Une erreur est survenue (Elec): {e}")
            print(f"Erreur Simulation Électrique: {type(e).__name__}: {e}")
            self.status_var.set("Erreur de simulation électrique.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()