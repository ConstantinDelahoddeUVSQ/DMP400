import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.abspath(__file__))
# Utiliser os.path.join pour la compatibilité multi-OS
path_partie_bleue = os.path.join(folder, "..", "SIMS", "Partie Bleue (accélération)", "Code")
path_partie_verte = os.path.join(folder, "..", "SIMS", "Partie Verte (déviation magnétique)", "Code")
# Ajouter aussi le dossier courant si deviation.py s'y trouve
sys.path.append(folder)
sys.path.append(path_partie_bleue)
sys.path.append(path_partie_verte)

# --- Importations des modules de simulation ---
try:
    import deviation # type: ignore
    import partie_electroaimant # type: ignore
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Impossible d'importer les modules de simulation.")
    print(f"Vérifiez l'existence des fichiers .py dans:")
    print(f"  '{folder}' (pour deviation.py)")
    print(f"  '{path_partie_bleue}'")
    print(f"  '{path_partie_verte}'")
    print("Assurez-vous que ces dossiers sont corrects et contiennent les fichiers __init__.py si nécessaire.")
    sys.exit(1)

# --- Classe principale de l'application ---
class ParticleApp:
    def __init__(self, root):
        """
        Initialise l'application de simulation SIMS.

        Parameters
        ----------
        root : tk.Tk
            La fenêtre racine de l'application Tkinter.
        """
        self.root = root
        self.root.title("Simulateur SIMS - Déviations") # Titre mis à jour
        # Ajustement taille initiale pour mieux voir le panneau de contrôle
        self.root.geometry("1400x700")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Style ---
        style = ttk.Style()
        # Essayer d'utiliser un thème natif si possible, sinon 'clam'
        try:
            style.theme_use('vista') # Windows
        except tk.TclError:
            try:
                style.theme_use('aqua') # macOS
            except tk.TclError:
                style.theme_use('clam') # Fallback
        style.configure("TButton", padding=6, relief="flat")
        # Augmenter la taille de la police pour les titres de section
        style.configure("TLabelframe.Label", font=('Helvetica', 13, 'bold'))
        style.configure("TLabel", padding=2)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        # Style pour boutons du tableau périodique (optionnel)
        style.configure("Element.TButton", padding=2, font=('Segoe UI', 9))
        style.configure("LanAct.TButton", padding=2, font=('Segoe UI', 9), background="#e8f4ea") # Vert pâle

        # --- Données ---
        self.particles_data = [] # Liste de tuples (nom: str, masse_u: float, charge_e: float)
        # ADDED: Variable pour stocker l'index de la particule sélectionnée pour la comparaison de potentiel
        self.selected_potential_particle_info = None # Sera un tuple (index, nom, masse, charge)

        # --- Structure Principale (PanedWindow) ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Panneau de Contrôle Scrollable (Gauche) ---
        # Conteneur pour le Canvas et la Scrollbar
        container_frame = ttk.Frame(main_paned_window, width=450) # Donner une largeur initiale
        container_frame.pack_propagate(False) # Empêcher le frame de rétrécir
        main_paned_window.add(container_frame, weight=0) # Poids 0 pour largeur fixe initiale

        # Canvas pour le contenu scrollable
        self.control_canvas = tk.Canvas(container_frame)
        # Scrollbar liée au Canvas
        self.scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=self.control_canvas.yview)
        # Frame interne qui contiendra tous les widgets de contrôle
        self.scrollable_frame = ttk.Frame(self.control_canvas)

        # Configurer la scrollregion quand le contenu change de taille
        self.scrollable_frame.bind("<Configure>", lambda e: self._update_scroll_region_and_bar(e))

        # Placer le frame interne dans le Canvas
        self.window_id = self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Lier la scrollbar au Canvas
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Placement du Canvas et de la Scrollbar dans leur conteneur
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Initialement cachée, sera affichée par _update_scrollbar_state si besoin
        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Ajuster la largeur du contenu interne quand le Canvas est redimensionné
        # Et mettre à jour l'état de la scrollbar
        self.control_canvas.bind("<Configure>", self._resize_canvas_content_and_update_bar)

        # --- Liaisons Molette Souris pour le défilement ---
        self.control_canvas.bind("<Enter>", lambda e: self._bind_mousewheel(True))
        self.control_canvas.bind("<Leave>", lambda e: self._bind_mousewheel(False))

        control_panel = self.scrollable_frame

        # --- Widgets dans le Panneau de Contrôle ---
        # Section Particules
        particle_frame = ttk.LabelFrame(control_panel, text="Gestion des Particules")
        particle_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_panel.columnconfigure(0, weight=1) # Permettre au frame de s'étendre en largeur

        self.create_particle_widgets(particle_frame)

        # Section Onglets Simulations
        self.notebook = ttk.Notebook(control_panel)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        control_panel.rowconfigure(1, weight=1) # Permettre au notebook de s'étendre

        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)
        self.pot_tab = ttk.Frame(self.notebook) # MODIFIED: Nom inchangé, contenu modifié

        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')
        self.notebook.add(self.pot_tab, text='Déviation Électrique (potentiel)') # MODIFIED: Nom inchangé

        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)
        self.create_pot_widgets(self.pot_tab) # MODIFIED: Appel de la fonction modifiée

        # --- Panneau Plot (Droite) ---
        plot_panel = ttk.Frame(main_paned_window)
        main_paned_window.add(plot_panel, weight=1) # Poids > 0 pour qu'il prenne le reste

        # Zone Matplotlib
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout(pad=3.0) # Ajouter un peu d'espace
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Barre d'outils Matplotlib
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Barre de Statut ---
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Mettre à jour la scrollbar une fois que tout est dessiné
        self.root.after(100, self._update_scrollbar_state)

    # --- Fonctions de Gestion Scrollbar ---
    def _bind_mousewheel(self, enter):
        """Lie ou délie les événements de molette pour le canvas."""
        platform = self.root.tk.call('tk', 'windowingsystem')
        if platform == 'win32':
            self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add=enter)
        elif platform == 'x11': # Linux
            self.control_canvas.bind_all("<Button-4>", self._on_mousewheel, add=enter)
            self.control_canvas.bind_all("<Button-5>", self._on_mousewheel, add=enter)
        else: # macOS ('aqua')
             self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add=enter)

    def _update_scroll_region_and_bar(self, event=None):
        """Met à jour la scrollregion ET l'état de la scrollbar."""
        # Utiliser after_idle pour s'assurer que les dimensions sont finales
        self.root.after_idle(self._perform_scroll_update)

    def _perform_scroll_update(self):
        try:
            if self.control_canvas.winfo_exists():
                self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
                self._update_scrollbar_state()
        except tk.TclError:
            pass # Évite erreur si la fenêtre se ferme

    def _resize_canvas_content_and_update_bar(self, event=None):
        """Redimensionne le contenu interne et met à jour l'état de la scrollbar."""
        try:
             if self.control_canvas.winfo_exists():
                canvas_width = event.width
                self.control_canvas.itemconfig(self.window_id, width=canvas_width)
                # Mettre à jour aussi la scrollregion ici car la largeur peut affecter la hauteur
                self._perform_scroll_update()
        except tk.TclError:
            pass # Évite erreur si la fenêtre se ferme

    def _update_scrollbar_state(self):
        """Active ou désactive la scrollbar si le contenu dépasse."""
        # Utiliser after_idle pour laisser le temps à Tkinter de calculer les tailles
        self.root.after_idle(self._check_and_set_scrollbar_state)

    def _check_and_set_scrollbar_state(self):
        """Vérifie et active/désactive la scrollbar."""
        try:
            if self.control_canvas.winfo_exists() and self.scrollable_frame.winfo_exists():
                canvas_height = self.control_canvas.winfo_height()
                content_height = self.scrollable_frame.winfo_reqheight()
                # print(f"Canvas Height: {canvas_height}, Content Height: {content_height}") # Debug
                if content_height <= canvas_height:
                    # Cacher la scrollbar si pas nécessaire
                    if self.scrollbar.winfo_ismapped():
                        self.scrollbar.pack_forget()
                else:
                    # Afficher la scrollbar si nécessaire
                    if not self.scrollbar.winfo_ismapped():
                        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except tk.TclError:
            # print("TclError in _check_and_set_scrollbar_state") # Debug
            pass # Fenêtre en cours de fermeture

    def _on_mousewheel(self, event):
        """Gère le défilement par molette si scroll nécessaire."""
        try:
            if not self.control_canvas.winfo_exists() or not self.scrollable_frame.winfo_exists():
                return # Ne rien faire si widgets détruits

            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height:
                return # Pas besoin de scroller

            # Défilement différent selon plateforme/type d'event
            delta = 0
            if event.num == 5 or event.delta < 0: # Windows/Mac scroll down, Linux button 5
                delta = 1
            elif event.num == 4 or event.delta > 0: # Windows/Mac scroll up, Linux button 4
                delta = -1

            if delta != 0:
                self.control_canvas.yview_scroll(delta, "units")
                return "break" # Empêcher la propagation si on a scrollé ici

        except tk.TclError:
             # print("TclError in _on_mousewheel") # Debug
             return # Erreur si fenêtre fermée pendant l'event
        except Exception as e:
            # print(f"Unexpected error in _on_mousewheel: {e}") # Debug
            return


    # --- Fermeture Propre ---
    def _on_closing(self):
        """Gère la fermeture de la fenêtre."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le simulateur ?"):
            try:
                plt.close(self.fig)
            except Exception as e:
                print(f"Erreur fermeture Matplotlib: {e}")
            try:
                # Important: annuler les callbacks 'after' pour éviter les erreurs TclError
                for after_id in self.root.tk.eval('after info').split():
                    self.root.after_cancel(after_id)
                self.root.destroy()
            except Exception as e:
                print(f"Erreur destruction Tkinter: {e}")

    # --- Widgets Section Particules ---
    def create_particle_widgets(self, parent):
        """
        Crée les widgets pour ajouter, lister et supprimer des particules.
        MODIFIED: Stocke aussi le nom dans self.particles_data.
        """
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill=tk.X)

        input_frame.columnconfigure(1, weight=1) # Colonne de l'Entry "Masse"
        input_frame.columnconfigure(3, weight=1) # Colonne de l'Entry "Charge"

        ttk.Label(input_frame, text="Masse (u):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mass_entry = ttk.Entry(input_frame, width=10)
        self.mass_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.mass_entry.insert(0, "1.0")

        ttk.Label(input_frame, text="Charge (e):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.charge_entry = ttk.Entry(input_frame, width=10)
        self.charge_entry.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        self.charge_entry.insert(0, "1.0")

        add_btn = ttk.Button(input_frame, text="Ajouter", command=self.add_particle)
        add_btn.grid(row=0, column=4, padx=10, pady=2)

        # --- Section Raccourcis ---
        ttk.Label(input_frame, text="Raccourcis :").grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))

        btns_frame = ttk.Frame(input_frame)
        btns_frame.grid(row=2, column=0, columnspan=5, pady=5, sticky="ew")

        num_btns = 3
        for i in range(num_btns):
            btns_frame.columnconfigure(i, weight=1)

        btn_o2 = ttk.Button(btns_frame, text="O₂⁻", command=lambda: self.ajt_particle_connue(31.998, -1.0))
        btn_o2.grid(row=0, column=0, padx=2, sticky="ew")

        btn_si = ttk.Button(btns_frame, text="Si⁺", command=lambda: self.ajt_particle_connue(28.085, +1.0))
        btn_si.grid(row=0, column=1, padx=2, sticky="ew")

        btn_h = ttk.Button(btns_frame, text="H⁺", command=lambda: self.ajt_particle_connue(1.008, +1.0))
        btn_h.grid(row=0, column=2, padx=2, sticky="ew")

        create_molecule_btn = ttk.Button(parent, text="Construire une Particule...", command=self.ouvrir_fenetre_tp)
        create_molecule_btn.pack(pady=(5, 10), padx=10, fill=tk.X)

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True, ipady=10)

        self.particle_tree = ttk.Treeview(tree_frame, columns=('Name', 'Mass (u)', 'Charge (e)'), show='headings', height=6)
        self.particle_tree.heading('Name', text='Nom')
        self.particle_tree.column('Name', width=120, anchor=tk.W)
        self.particle_tree.heading('Mass (u)', text='Masse (u)')
        self.particle_tree.column('Mass (u)', width=100, anchor=tk.CENTER)
        self.particle_tree.heading('Charge (e)', text='Charge (e)')
        self.particle_tree.column('Charge (e)', width=100, anchor=tk.CENTER)

        scrollbar_tree = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar_tree.set)
        self.particle_tree.bind("<MouseWheel>", lambda e: self._on_mousewheel(e))

        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer Sélection", command=self.remove_particle)
        remove_btn.pack(pady=5, padx=10, fill=tk.X)

    # --- Fenêtre Tableau Périodique ---
    # (Fonctions ouvrir_fenetre_tp, construction_de_molecule, reset_molecule, _update_molecule_display, submit_molecule restent inchangées)
    def ouvrir_fenetre_tp(self):
        """
        Ouvre une fenêtre modale pour construire une molécule/ion.
        """
        # Empêcher l'ouverture multiple
        if hasattr(self, 'molecule_fenetre') and self.molecule_fenetre.winfo_exists():
            self.molecule_fenetre.lift()
            return

        self.molecule_fenetre = tk.Toplevel(self.root)
        self.molecule_fenetre.title("Construire une Particule")
        # Ajuster la taille pour bien afficher le tableau
        self.molecule_fenetre.geometry("1100x650")
        self.molecule_fenetre.grab_set()
        self.molecule_fenetre.transient(self.root)

        # Stockage de la molécule en cours
        self.selected_elts = {} # Dictionnaire: {'symbole': {'mass': float, 'count': int}}

        # --- Tableau Périodique ---
        periodic_layout = [
            [('H', 1.008), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ('He', 4.0026)],
            [('Li', 6.94), ('Be', 9.0122), None, None, None, None, None, None, None, None, None, None, ('B', 10.81), ('C', 12.011), ('N', 14.007), ('O', 15.999), ('F', 18.998), ('Ne', 20.180)],
            [('Na', 22.990), ('Mg', 24.305), None, None, None, None, None, None, None, None, None, None, ('Al', 26.982), ('Si', 28.085), ('P', 30.974), ('S', 32.06), ('Cl', 35.45), ('Ar', 39.948)],
            [('K', 39.098), ('Ca', 40.078), ('Sc', 44.956), ('Ti', 47.867), ('V', 50.942), ('Cr', 51.996), ('Mn', 54.938), ('Fe', 55.845), ('Co', 58.933), ('Ni', 58.693), ('Cu', 63.546), ('Zn', 65.38), ('Ga', 69.723), ('Ge', 72.630), ('As', 74.922), ('Se', 78.971), ('Br', 79.904), ('Kr', 83.798)],
            [('Rb', 85.468), ('Sr', 87.62), ('Y', 88.906), ('Zr', 91.224), ('Nb', 92.906), ('Mo', 95.95), ('Tc', 98.0), ('Ru', 101.07), ('Rh', 102.91), ('Pd', 106.42), ('Ag', 107.87), ('Cd', 112.41), ('In', 114.82), ('Sn', 118.71), ('Sb', 121.76), ('Te', 127.60), ('I', 126.90), ('Xe', 131.29)],
            [('Cs', 132.91), ('Ba', 137.33), ('La', 138.91), ('Hf', 178.49), ('Ta', 180.95), ('W', 183.84), ('Re', 186.21), ('Os', 190.23), ('Ir', 192.22), ('Pt', 195.08), ('Au', 196.97), ('Hg', 200.59), ('Tl', 204.38), ('Pb', 207.2), ('Bi', 208.98), ('Po', 209.0), ('At', 210.0), ('Rn', 222.0)],
            [('Fr', 223.0), ('Ra', 226.0), ('Ac', 227.0), ('Rf', 267.0), ('Db', 270.0), ('Sg', 271.0), ('Bh', 270.0), ('Hs', 277.0), ('Mt', 278.0), ('Ds', 281.0), ('Rg', 282.0), ('Cn', 285.0), ('Nh', 286.0), ('Fl', 289.0), ('Mc', 290.0), ('Lv', 293.0), ('Ts', 294.0), ('Og', 294.0)],
            [], # Ligne vide pour espacement
            [None, None, None, ('Ce', 140.12), ('Pr', 140.91), ('Nd', 144.24), ('Pm', 145.0), ('Sm', 150.36), ('Eu', 151.96), ('Gd', 157.25), ('Tb', 158.93), ('Dy', 162.50), ('Ho', 164.93), ('Er', 167.26), ('Tm', 168.93), ('Yb', 173.05), ('Lu', 174.97), None], # Lanthanides décalés
            [None, None, None, ('Th', 232.04), ('Pa', 231.04), ('U', 238.03), ('Np', 237.0), ('Pu', 244.0), ('Am', 243.0), ('Cm', 247.0), ('Bk', 247.0), ('Cf', 251.0), ('Es', 252.0), ('Fm', 257.0), ('Md', 258.0), ('No', 259.0), ('Lr', 262.0), None] # Actinides décalés
        ]

        table_frame = ttk.Frame(self.molecule_fenetre)
        table_frame.pack(pady=10, padx=10)

        # Création des boutons
        for row_idx, row in enumerate(periodic_layout):
            pady_val = 5 if row_idx == 8 else 2 # Espacement avant La/Ac
            for col_idx, element in enumerate(row):
                if element:
                    symbol, mass = element
                    is_lan_act_row = row_idx >= 8
                    # Utiliser le style défini précédemment ou 'TButton' par défaut
                    btn_style = "LanAct.TButton" if is_lan_act_row else "Element.TButton"

                    btn = ttk.Button(table_frame, text=symbol, width=4, style=btn_style,
                                     command=lambda s=symbol, m=mass: self.construction_de_molecule(s, m))
                    btn.grid(row=row_idx, column=col_idx, padx=1, pady=pady_val, sticky="nsew")

        # --- Affichage Molécule en cours et Contrôles ---
        control_frame = ttk.Frame(self.molecule_fenetre)
        control_frame.pack(pady=10, padx=20, fill=tk.X)

        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=0)
        control_frame.columnconfigure(2, weight=1)

        ttk.Label(control_frame, text="Particule construite:", font=('Helvetica', 14, 'bold')).grid(row=0, column=1, sticky="", pady=(0,5))

        display_reset_frame = ttk.Frame(control_frame)
        display_reset_frame.grid(row=1, column=1, sticky="ew", pady=5)
        display_reset_frame.columnconfigure(0, weight=1)

        self.molecule_display_var = tk.StringVar(value="(vide)")
        display_label = ttk.Label(display_reset_frame, textvariable=self.molecule_display_var,
                                   relief=tk.SUNKEN, padding=5, anchor=tk.W, font=('Consolas', 10))
        display_label.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))

        reset_btn = ttk.Button(display_reset_frame, text="Effacer", command=self.reset_molecule)
        reset_btn.grid(row=0, column=1, sticky=tk.E)

        charge_frame = ttk.Frame(control_frame)
        charge_frame.grid(row=2, column=1, sticky="", pady=5)

        charge_label = ttk.Label(charge_frame, text="Charge (e):")
        charge_label.pack(side=tk.LEFT)
        self.molecule_charge_var = tk.StringVar(value="1")
        charge_entry = ttk.Entry(charge_frame, textvariable=self.molecule_charge_var, width=8)
        charge_entry.pack(side=tk.LEFT, padx=5)

        submit_btn = ttk.Button(control_frame, text="Ajouter cette Particule à la liste", command=self.submit_molecule)
        submit_btn.grid(row=3, column=1, sticky="", pady=10)

    def construction_de_molecule(self, symbol, mass):
        """Ajoute ou incrémente un élément dans la molécule en cours."""
        if symbol in self.selected_elts:
            self.selected_elts[symbol]['count'] += 1
        else:
            self.selected_elts[symbol] = {'mass': mass, 'count': 1}
        self._update_molecule_display()

    def reset_molecule(self):
        """Réinitialise la molécule en cours de construction."""
        self.selected_elts = {}
        self._update_molecule_display()

    def _update_molecule_display(self):
        """Met à jour l'affichage de la formule moléculaire."""
        if not self.selected_elts:
            self.molecule_display_var.set("(vide)")
            return

        sorted_symbols = sorted(self.selected_elts.keys())
        molecule_parts = []
        for symbol in sorted_symbols:
            data = self.selected_elts[symbol]
            count = data['count']
            part = symbol
            if count > 1:
                subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                part += str(count).translate(subscript_map)
            molecule_parts.append(part)
        self.molecule_display_var.set("".join(molecule_parts))

    def ajt_particle_connue(self, mass_u, charge_e):
        """Ajoute une particule prédéfinie (O₂⁻, Si⁺, H⁺) avec un nom explicite."""
        symbole = ""
        if abs(mass_u - 31.998) < 1e-3: symbole = "O₂"
        elif abs(mass_u - 28.085) < 1e-3: symbole = "Si"
        elif abs(mass_u - 1.008) < 1e-3: symbole = "H"
        else: symbole = f"{mass_u:.3f}u"

        charge_sign = '+' if charge_e > 0 else '-'
        charge_val = abs(int(charge_e))
        charge_str = f"({charge_val}{charge_sign})" if charge_val != 1 else charge_sign
        nom = f"{symbole}{charge_str}"

        self._add_particle_to_list(mass_u, charge_e, nom)

    def submit_molecule(self):
        """Calcule la masse, lit la charge, et ajoute la particule construite."""
        if not self.selected_elts:
            messagebox.showwarning("Aucun Élément", "Veuillez cliquer sur les éléments du tableau pour construire votre particule.", parent=self.molecule_fenetre)
            return

        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())
            charge_str = self.molecule_charge_var.get().strip().replace(',', '.')
            if not charge_str: raise ValueError("La charge ne peut pas être vide.")
            charge = float(charge_str)

            formula = self.molecule_display_var.get()
            charge_sign = '+' if charge > 0 else '-'
            charge_val = abs(int(charge))
            charge_disp_str = f"({charge_val}{charge_sign})" if charge_val != 1 or charge == 0 else charge_sign
            nom = f"{formula}{charge_disp_str}"

            added = self._add_particle_to_list(total_mass, charge, nom)
            if added:
                self.molecule_fenetre.destroy()

        except ValueError as e:
            messagebox.showerror("Erreur de Saisie", f"Erreur de soumission : {e}", parent=self.molecule_fenetre)
            self.status_var.set("Erreur lors de la soumission de molécule.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur inattendue est survenue: {e}", parent=self.molecule_fenetre)
            self.status_var.set("Erreur lors de la soumission de molécule.")

    # --- Logique d'ajout et de suppression ---
    def add_particle(self):
        """Ajoute une particule depuis les champs d'entrée manuels."""
        try:
            mass_u_str = self.mass_entry.get().strip().replace(',', '.')
            charge_e_str = self.charge_entry.get().strip().replace(',', '.')
            if not mass_u_str or not charge_e_str: raise ValueError("Masse et Charge ne peuvent être vides.")
            mass_u = float(mass_u_str)
            charge_e = float(charge_e_str)

            # Générer un nom simple pour l'ajout manuel
            charge_sign = '+' if charge_e > 0 else '-'
            charge_val = abs(float(f"{charge_e:.2f}")) # Format pour éviter trop de décimales
            nom = f"M={mass_u:.2f} C={charge_val}{charge_sign}"

            added = self._add_particle_to_list(mass_u, charge_e, nom)
            if added:
                self.mass_entry.delete(0, tk.END)
                self.charge_entry.delete(0, tk.END)

        except ValueError as e:
            messagebox.showerror("Erreur d'Entrée", f"Entrée invalide : {e}")
            self.status_var.set("Erreur d'ajout de particule.")
        except Exception as e:
             messagebox.showerror("Erreur", f"Une erreur inattendue est survenue: {e}")
             self.status_var.set("Erreur d'ajout de particule.")

    def _add_particle_to_list(self, mass_u, charge_e, name):
        """
        Fonction interne pour valider et ajouter une particule à la liste et au Treeview.
        Retourne True si ajoutée, False sinon.
        MODIFIED: Stocke (name, mass_u, charge_e).
        """
        try:
            if mass_u <= 0: raise ValueError("Masse doit être > 0.")
            if charge_e == 0: raise ValueError("Charge ne peut pas être nulle pour la déviation.")

            current_sign = np.sign(charge_e)
            if self.particles_data: # Vérifier seulement s'il y a déjà des particules
                existing_sign = np.sign(self.particles_data[0][2]) # Index 2 pour charge_e
                if current_sign != existing_sign:
                    raise ValueError("Toutes les particules doivent avoir des charges de même signe.")

            # Arrondir pour comparaison de doublon (basé sur masse et charge)
            particle_key = (round(mass_u, 5), round(charge_e, 5))
            existing_keys = [(round(p[1], 5), round(p[2], 5)) for p in self.particles_data]

            if particle_key not in existing_keys:
                # Ajouter (nom, masse, charge)
                self.particles_data.append((name, mass_u, charge_e))
                self.particle_tree.insert('', tk.END, values=(name, f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                self.status_var.set(f"Particule '{name}' ajoutée.")
                self._update_scroll_region_and_bar() # Mettre à jour si liste s'allonge
                return True
            else:
                parent_window = getattr(self, 'molecule_fenetre', self.root)
                if parent_window.winfo_exists():
                    messagebox.showwarning("Doublon", f"La particule avec masse {mass_u:.3f} u et charge {charge_e:+.2f} e est déjà dans la liste.", parent=parent_window)
                else:
                     messagebox.showwarning("Doublon", f"La particule avec masse {mass_u:.3f} u et charge {charge_e:+.2f} e est déjà dans la liste.")
                self.status_var.set("Ajout annulé (doublon).")
                return False

        except ValueError as e:
            parent_window = getattr(self, 'molecule_fenetre', self.root)
            if parent_window.winfo_exists(): messagebox.showerror("Erreur de Validation", f"{e}", parent=parent_window)
            else: messagebox.showerror("Erreur de Validation", f"{e}")
            self.status_var.set(f"Erreur validation: {e}")
            return False
        except Exception as e:
             parent_window = getattr(self, 'molecule_fenetre', self.root)
             if parent_window.winfo_exists(): messagebox.showerror("Erreur Inattendue", f"{e}", parent=parent_window)
             else: messagebox.showerror("Erreur Inattendue", f"{e}")
             self.status_var.set("Erreur interne lors de l'ajout.")
             return False

    def remove_particle(self):
        """Supprime la ou les particules sélectionnées dans le Treeview."""
        selected_items = self.particle_tree.selection()
        if not selected_items:
            messagebox.showwarning("Aucune Sélection", "Veuillez sélectionner une ou plusieurs particules à supprimer dans la liste.")
            return

        if len(selected_items) > 1:
            if not messagebox.askyesno("Confirmation", f"Supprimer les {len(selected_items)} particules sélectionnées ?"):
                return

        indices_to_remove = []
        items_to_remove_tree = []
        names_removed = []

        for item_id in selected_items:
            try:
                index = self.particle_tree.index(item_id)
                indices_to_remove.append(index)
                items_to_remove_tree.append(item_id)
                names_removed.append(self.particle_tree.item(item_id, 'values')[0])
            except tk.TclError:
                print(f"Avertissement: Impossible de trouver l'index pour l'item {item_id}")

        indices_to_remove.sort(reverse=True)
        deleted_count = 0
        for index in indices_to_remove:
            try:
                # Si la particule supprimée est celle sélectionnée pour la comparaison, réinitialiser
                if self.selected_potential_particle_info and self.selected_potential_particle_info[0] == index:
                    self._reset_potential_comparison_state()

                del self.particles_data[index]
                deleted_count += 1

                # ADDED: Décaler les index stockés si nécessaire
                if self.selected_potential_particle_info and self.selected_potential_particle_info[0] > index:
                     # Décrémenter l'index stocké car un élément avant lui a été supprimé
                     old_info = self.selected_potential_particle_info
                     self.selected_potential_particle_info = (old_info[0] - 1, old_info[1], old_info[2], old_info[3])

            except IndexError:
                 print(f"Avertissement: Index {index} hors limites pour self.particles_data.")

        for item_id in items_to_remove_tree:
            if self.particle_tree.exists(item_id):
                self.particle_tree.delete(item_id)

        self.status_var.set(f"{deleted_count} particule(s) supprimée(s): {', '.join(names_removed)}.")
        self._update_scroll_region_and_bar() # Mettre à jour si liste raccourcit


    # --- Widgets Magnétiques (Inchangé) ---
    def create_magnetic_widgets(self, parent):
        """
        Crée les widgets pour la simulation de déviation magnétique.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Frames pour afficher/cacher selon le mode dynamique
        self.dynamic_inputs_frame = ttk.Frame(frame)
        self.base_inputs_frame = ttk.Frame(frame)

        # Entrée commune : X détecteur
        self.x_detecteur_var = tk.StringVar(value="0.1")
        self.add_labeled_entry(frame, "X détecteur (m):", self.x_detecteur_var).pack(fill=tk.X, pady=3)

        # Checkbox pour choisir le mode
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)",
                                        variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs)
        dynamic_check.pack(anchor=tk.W, pady=5)

        # --- Widgets Mode Non-Dynamique (Statique) ---
        parent_base = self.base_inputs_frame
        self.v0_mag_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)
        self.bz_mag_var = tk.StringVar(value="0.2")
        self.add_labeled_entry(parent_base, "Champ Magnétique (T):", self.bz_mag_var).pack(fill=tk.X, pady=3)
        # Bouton pour lancer la simulation statique
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_magnetic_simulation)
        trace_btn_base.pack(pady=15)

        # --- Widgets Mode Dynamique ---
        parent_dyn = self.dynamic_inputs_frame
        # Limites pour Bz
        self.bz_min_var = tk.StringVar(value="0.01")
        self.add_labeled_entry(parent_dyn, "Bz min (T):", self.bz_min_var).pack(fill=tk.X, pady=3)
        self.bz_max_var = tk.StringVar(value="0.5")
        self.add_labeled_entry(parent_dyn, "Bz max (T):", self.bz_max_var).pack(fill=tk.X, pady=3)

        # Slider Bz
        ttk.Label(parent_dyn, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_bz = ttk.Frame(parent_dyn)
        self.slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=0.2)
        self.bz_slider = ttk.Scale(self.slider_frame_bz, from_=0.01, to=0.5, orient=tk.HORIZONTAL,
                                   variable=self.bz_var, command=self._on_bz_slider_change)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(self.slider_frame_bz, textvariable=self.bz_label_var, width=10).pack(side=tk.LEFT)

        # Limites pour V0
        self.v0_min_var = tk.StringVar(value="1e5")
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.v0_min_var).pack(fill=tk.X, pady=3)
        self.v0_max_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.v0_max_var).pack(fill=tk.X, pady=3)

        # Slider V0
        ttk.Label(parent_dyn, text="Vitesse Initiale (m/s):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_v0 = ttk.Frame(parent_dyn)
        self.slider_frame_v0.pack(fill=tk.X, pady=(0,5))
        self.v0_var = tk.DoubleVar(value=5.5e5)
        self.v0_slider = ttk.Scale(self.slider_frame_v0, from_=1e5, to=1e6, orient=tk.HORIZONTAL,
                                   variable=self.v0_var, command=self._on_v0_slider_change)
        self.v0_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var = tk.StringVar(value=f"{self.v0_var.get():.2e} m/s")
        ttk.Label(self.slider_frame_v0, textvariable=self.v0_label_var, width=12).pack(side=tk.LEFT)

        # Bouton pour appliquer les limites et tracer initialement en mode dynamique
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_magnetic_simulation)
        apply_limits_btn_dyn.pack(pady=15)

        # Afficher le bon groupe de widgets au départ
        self.toggle_dynamic_inputs()

    def toggle_dynamic_inputs(self) :
        """Gère l'affichage des widgets magnétiques selon le mode choisi."""
        if self.dynamic_trace_var.get():
            self.base_inputs_frame.pack_forget()
            self.dynamic_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else:
            self.dynamic_inputs_frame.pack_forget()
            self.base_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        self._update_scroll_region_and_bar() # Mettre à jour scroll

    def _on_bz_slider_change(self, event=None):
        self._update_bz_label()
        if self.particles_data: self.run_magnetic_simulation(called_by_slider=True)
    def _update_bz_label(self, event=None): self.bz_label_var.set(f"{self.bz_var.get():.3f} T")
    def _on_v0_slider_change(self, event=None):
        self._update_v0_label()
        if self.particles_data: self.run_magnetic_simulation(called_by_slider=True)
    def _update_v0_label(self, event=None): self.v0_label_var.set(f"{self.v0_var.get():.2e} m/s")


    # --- Widgets Électriques (Inchangé) ---
    def create_electric_widgets(self, parent):
        """
        Crée les widgets pour la simulation de déviation électrique, y compris l'incertitude.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Paramètres Communs ---
        self.angle_var = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3)
        self.dist_var = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3)

        # --- 2. Checkbox Incertitudes ---
        self.show_uncertainty_var = tk.BooleanVar(value=False)
        self.uncertainty_check = ttk.Checkbutton(frame, text="Afficher Incertitudes", variable=self.show_uncertainty_var, command=self.toggle_uncertainty_inputs)
        self.uncertainty_check.pack(anchor=tk.W, padx=5, pady=(5, 0))

        # --- 3. Frame Incertitudes (créé mais packé par toggle) ---
        self.uncertainty_inputs_frame = ttk.LabelFrame(frame, text="Paramètres d'incertitude (%)")
        self.delta_v0_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔV0/V0 (%):", self.delta_v0_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_theta_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "Δθ/θ (% de l'angle):", self.delta_theta_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_h_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "Δh/h (%):", self.delta_h_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_E_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔE/E (%):", self.delta_E_percent_var).pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(self.uncertainty_inputs_frame, text="Note: Δm/m et Δq/q utiliseront des valeurs fixes (0.1%)", font=('Segoe UI', 8)).pack(pady=(5,0))

        # --- 4. Séparateur ---
        self.elec_separator = ttk.Separator(frame, orient=tk.HORIZONTAL)
        # Note: Le pack du séparateur sera géré par toggle_uncertainty_inputs

        # --- 5. Frame Contrôle Mode ---
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        self.dynamic_elec_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)",
                                             variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric)
        # Note: Le pack du checkbox sera géré par toggle_uncertainty_inputs

        self.dynamic_electric_inputs_frame = ttk.Frame(frame)
        self.base_electric_inputs_frame = ttk.Frame(frame)

        # --- Widgets Non-Dynamiques (Base) ---
        parent_base = self.base_electric_inputs_frame
        self.v0_elec_var = tk.StringVar(value="1e5") # m/s
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)
        self.diff_pot_var = tk.StringVar(value="-5000") # Volts
        self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_electric_simulation)
        trace_btn_base.pack(pady=15)

        # --- Widgets Mode Dynamique ---
        parent_dyn = self.dynamic_electric_inputs_frame
        self.elec_v0_min_var = tk.StringVar(value="1e4")
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.elec_v0_min_var).pack(fill=tk.X, pady=3)
        self.elec_v0_max_var = tk.StringVar(value="2e5")
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.elec_v0_max_var).pack(fill=tk.X, pady=3)

        ttk.Label(parent_dyn, text="Vitesse Initiale V0 (m/s):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v0_elec = ttk.Frame(parent_dyn)
        self.slider_frame_v0_elec.pack(fill=tk.X, pady=(0, 5))
        self.v0_var_elec = tk.DoubleVar(value=1e5)
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=1e4, to=2e5, orient=tk.HORIZONTAL,
                                        variable=self.v0_var_elec, command=self._on_v0_slider_change_elec)
        self.v0_slider_elec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var_elec = tk.StringVar(value=f"{self.v0_var_elec.get():.2e} m/s")
        ttk.Label(self.slider_frame_v0_elec, textvariable=self.v0_label_var_elec, width=12).pack(side=tk.LEFT)

        self.diff_pot_min_var = tk.StringVar(value="-10000")
        self.add_labeled_entry(parent_dyn, "Potentiel min (V):", self.diff_pot_min_var).pack(fill=tk.X, pady=3)
        self.diff_pot_max_var = tk.StringVar(value="10000")
        self.add_labeled_entry(parent_dyn, "Potentiel max (V):", self.diff_pot_max_var).pack(fill=tk.X, pady=3)

        ttk.Label(parent_dyn, text="Diff. Potentiel (V):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v = ttk.Frame(parent_dyn)
        self.slider_frame_v.pack(fill=tk.X, pady=(0, 5))
        self.pot_var = tk.DoubleVar(value=-5000)
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL,
                                    variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V")
        ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=12).pack(side=tk.LEFT)

        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_electric_simulation)
        apply_limits_btn_dyn.pack(pady=15)

        # --- Afficher/Cacher initialement ---
        self.toggle_uncertainty_inputs() # Gère le pack initial du frame incertitude, du séparateur et du checkbox dynamique
        self.toggle_dynamic_electric() # Gère le pack initial du frame statique ou dynamique

    def toggle_dynamic_electric(self):
        """Gère l'affichage des widgets électriques dynamique/statique."""
        if self.dynamic_elec_var.get():
            self.base_electric_inputs_frame.pack_forget()
            # Pack le frame dynamique APRES le checkbox correspondant
            self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        else:
            self.dynamic_electric_inputs_frame.pack_forget()
            # Pack le frame statique APRES le checkbox correspondant
            self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        self._update_scroll_region_and_bar() # Mettre à jour scroll

    def toggle_uncertainty_inputs(self):
        """Affiche ou cache le frame des entrées d'incertitude et gère le positionnement des éléments suivants."""
        if self.show_uncertainty_var.get():
            # Pack frame incertitude AVANT le séparateur
            self.uncertainty_inputs_frame.pack(fill=tk.X, pady=(5,0), padx=5, before=self.uncertainty_check) # Pack avant le check qui l'a déclenché
            # Afficher le séparateur APRES le frame incertitude
            self.elec_separator.pack(fill=tk.X, pady=10, padx=5, after=self.uncertainty_inputs_frame)
             # Afficher le check dynamique APRES le séparateur
            self.dynamic_elec_check.pack(anchor=tk.W, padx=5, pady=(5, 0), after=self.elec_separator)
        else:
            self.uncertainty_inputs_frame.pack_forget()
             # Afficher le séparateur APRES le check qui était avant lui
            self.elec_separator.pack(fill=tk.X, pady=10, padx=5, after=self.uncertainty_check)
            # Afficher le check dynamique APRES le séparateur
            self.dynamic_elec_check.pack(anchor=tk.W, padx=5, pady=(5, 0), after=self.elec_separator)

        # Important: Il faut re-packer les frames de base/dynamique car leur widget 'after' a peut-être changé
        self.toggle_dynamic_electric()
        # Mettre à jour la scrollregion car la hauteur change
        # self._update_scroll_region_and_bar() # Déjà appelé par toggle_dynamic_electric

    def _on_pot_slider_change(self, event=None):
        self._update_pot_label()
        if self.particles_data: self.run_electric_simulation(called_by_slider=True)
    def _on_v0_slider_change_elec(self, event=None):
        self._update_v0_label_elec()
        if self.particles_data: self.run_electric_simulation(called_by_slider=True)
    def _update_pot_label(self, event=None): self.pot_label_var.set(f"{self.pot_var.get():.1f} V")
    def _update_v0_label_elec(self, event=None): self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e} m/s")


    # --- Widgets Potentiel (MODIFIED) ---
    def create_pot_widgets(self, parent):
        """
        Crée les widgets pour la comparaison de potentiels pour une particule sélectionnée.
        MODIFIED: Structure et fonctionnalité entièrement revues.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Paramètres Communs (Toujours visibles) ---
        self.angle_var_pot = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var_pot).pack(fill=tk.X, pady=3)

        self.dist_var_pot = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var_pot).pack(fill=tk.X, pady=3)

        self.v0_var_pot = tk.StringVar(value="1e5") # ADDED: Vitesse initiale nécessaire
        self.add_labeled_entry(frame, "Vitesse Initiale (m/s):", self.v0_var_pot).pack(fill=tk.X, pady=3)

        # --- 2. Bouton de Configuration ---
        self.configure_potential_button = ttk.Button(frame, text="Configurer Potentiels pour Sélection", command=self.setup_potential_comparison)
        self.configure_potential_button.pack(pady=(10, 5), fill=tk.X)

        # --- 3. Frame pour les entrées de potentiel (Caché initialement) ---
        self.potential_input_frame = ttk.Frame(frame, padding="5")
        # Ce frame sera packé par setup_potential_comparison

        # Label pour afficher la particule sélectionnée
        self.selected_particle_label = ttk.Label(self.potential_input_frame, text="Particule sélectionnée: (aucune)", relief=tk.GROOVE, padding=4)
        self.selected_particle_label.pack(fill=tk.X, pady=(0, 10))

        # Entrées pour les deux potentiels
        self.potential1_var = tk.StringVar(value="-5000") # ADDED
        self.add_labeled_entry(self.potential_input_frame, "Potentiel 1 (V):", self.potential1_var).pack(fill=tk.X, pady=3)

        self.potential2_var = tk.StringVar(value="-6000") # ADDED
        self.add_labeled_entry(self.potential_input_frame, "Potentiel 2 (V):", self.potential2_var).pack(fill=tk.X, pady=3)

        # Bouton pour lancer la simulation de comparaison
        self.trace_potential_button = ttk.Button(self.potential_input_frame, text="Tracer Simulation Comparaison", command=self.run_potential_comparison_simulation)
        self.trace_potential_button.pack(pady=15)

        # ADDED: Bouton Reset pour changer de particule ou cacher le frame
        reset_button = ttk.Button(self.potential_input_frame, text="Changer Particule / Réinitialiser", command=self._reset_potential_comparison_state)
        reset_button.pack(pady=(0, 5), fill=tk.X)

    # --- Helper --- (Inchangé)
    def add_labeled_entry(self, parent, label_text, string_var):
        """Crée une paire Label + Entry alignée."""
        entry_frame = ttk.Frame(parent)
        entry_frame.columnconfigure(0, weight=0)
        entry_frame.columnconfigure(1, weight=1)
        label = ttk.Label(entry_frame, text=label_text, anchor="w")
        label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.grid(row=0, column=1, sticky="ew", pady=2)
        return entry_frame

    # --- Exécution des Simulations ---

    # run_magnetic_simulation (Inchangé, juste adapté à la nouvelle structure de particles_data)
    def run_magnetic_simulation(self, called_by_slider=False):
        """Lance la simulation de déviation magnétique."""
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla(); self.ax.set_title("Déviation Magnétique"); self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("y (m)"); self.canvas.draw(); return

        try:
            x_detecteur_str = self.x_detecteur_var.get().strip().replace(',', '.')
            if not x_detecteur_str: raise ValueError("X détecteur ne peut être vide.")
            x_detecteur = float(x_detecteur_str)
            if x_detecteur <= 0: raise ValueError("X détecteur doit être positif.")

            if not self.dynamic_trace_var.get():
                v0_str = self.v0_mag_var.get().strip().replace(',', '.')
                bz_str = self.bz_mag_var.get().strip().replace(',', '.')
                if not v0_str or not bz_str: raise ValueError("V0 et Bz ne peuvent être vides.")
                v0 = float(v0_str); bz = float(bz_str)
                if v0 <= 0: raise ValueError("Vitesse initiale doit être > 0.")
                if bz == 0: raise ValueError("Champ magnétique ne peut être nul.")
            else:
                if not called_by_slider:
                    # ... (validation et mise à jour des sliders inchangées) ...
                    bz_min_str = self.bz_min_var.get().strip().replace(',', '.')
                    bz_max_str = self.bz_max_var.get().strip().replace(',', '.')
                    v0_min_str = self.v0_min_var.get().strip().replace(',', '.')
                    v0_max_str = self.v0_max_var.get().strip().replace(',', '.')
                    if not all([bz_min_str, bz_max_str, v0_min_str, v0_max_str]): raise ValueError("Les limites min/max ne peuvent être vides.")
                    bz_min = float(bz_min_str); bz_max = float(bz_max_str)
                    v0_min = float(v0_min_str); v0_max = float(v0_max_str)
                    if v0_min <= 0 : raise ValueError("V0 min doit être > 0.")
                    if v0_min >= v0_max : raise ValueError("V0 max doit être > V0 min.")
                    if bz_min == bz_max : raise ValueError("Bz max doit être != Bz min.") # Accepter négatifs ?
                    if bz_min > bz_max: bz_min, bz_max = bz_max, bz_min # Inverser si besoin

                    current_bz = self.bz_var.get(); current_v0 = self.v0_var.get()
                    self.bz_slider.config(from_=bz_min, to=bz_max)
                    if not (bz_min <= current_bz <= bz_max): self.bz_var.set((bz_max + bz_min) / 2)
                    self._update_bz_label()
                    self.v0_slider.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max): self.v0_var.set((v0_max + v0_min) / 2)
                    self._update_v0_label()

                v0 = self.v0_var.get(); bz = self.bz_var.get()
                if abs(bz) < 1e-15: raise ValueError("Le champ magnétique (Bz) est trop proche de zéro.")

            self.ax.cla()
            self.status_var.set("Calcul déviation magnétique...")
            self.root.update_idletasks()

            # MODIFIED: Extraire (masse, charge) de self.particles_data
            mass_charge_list_mag = [(p[1], p[2]) for p in self.particles_data]

            partie_electroaimant.tracer_ensemble_trajectoires(
                    mass_charge_list_mag, v0, bz, x_detecteur, create_plot=False, ax=self.ax
                )
            # MODIFIED: Ajouter les noms aux légendes si la fonction le permet ou manuellement
            handles, labels = self.ax.get_legend_handles_labels()
            new_labels = [f"{self.particles_data[i][0]} ({labels[i]})" for i in range(len(labels))]
            self.ax.legend(handles, new_labels, title="Particules")


            self.ax.relim(); self.ax.autoscale_view(True, True, True)
            self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")

        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Magnétique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Mag): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Une erreur est survenue (Magnétique):\n{e}", parent=self.root)
            print(f"Erreur Simulation Magnétique: {type(e).__name__}: {e}")
            self.status_var.set("Erreur simulation magnétique.")


    # run_electric_simulation (Inchangé, juste adapté à la nouvelle structure de particles_data)
    def run_electric_simulation(self, called_by_slider=False):
        """Lance la simulation de déviation électrique en utilisant deviation.py."""
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla(); self.ax.set_title("Déviation Électrique"); self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("y (m)"); self.canvas.draw(); return

        try:
            angle_deg_str = self.angle_var.get().strip().replace(',', '.')
            dist_str = self.dist_var.get().strip().replace(',', '.')
            if not angle_deg_str or not dist_str: raise ValueError("Angle et Distance/Hauteur requis.")
            angle_deg = float(angle_deg_str); hauteur_distance = float(dist_str)
            if hauteur_distance <= 0 : raise ValueError("Hauteur/Distance doit être > 0.")
            if not (0 < angle_deg < 90): raise ValueError("Angle doit être > 0° et < 90°.")
            angle_rad = np.radians(angle_deg); hauteur_initiale = hauteur_distance

            if not self.dynamic_elec_var.get():
                v0_str = self.v0_elec_var.get().strip().replace(',', '.')
                pot_str = self.diff_pot_var.get().strip().replace(',', '.')
                if not v0_str or not pot_str: raise ValueError("V0 et Potentiel requis.")
                v0 = float(v0_str); potentiel = float(pot_str)
                if v0 <= 0 : raise ValueError("Vitesse initiale doit être > 0.")
            else :
                if not called_by_slider :
                    # ... (validation et mise à jour sliders inchangées) ...
                    pot_min_str = self.diff_pot_min_var.get().strip().replace(',', '.')
                    pot_max_str = self.diff_pot_max_var.get().strip().replace(',', '.')
                    v0_min_str = self.elec_v0_min_var.get().strip().replace(',', '.')
                    v0_max_str = self.elec_v0_max_var.get().strip().replace(',', '.')
                    if not all([pot_min_str, pot_max_str, v0_min_str, v0_max_str]): raise ValueError("Les limites min/max ne peuvent être vides.")
                    pot_min = float(pot_min_str); pot_max = float(pot_max_str)
                    v0_min = float(v0_min_str); v0_max = float(v0_max_str)
                    if v0_min <= 0 : raise ValueError("V0 min doit être > 0.")
                    if v0_min >= v0_max : raise ValueError("V0 max doit être > V0 min.")
                    if pot_min >= pot_max : raise ValueError("Potentiel max doit être > Potentiel min.")

                    current_pot = self.pot_var.get(); current_v0 = self.v0_var_elec.get()
                    self.pot_slider.config(from_=pot_min, to=pot_max)
                    if not (pot_min <= current_pot <= pot_max): self.pot_var.set((pot_max + pot_min) / 2)
                    self._update_pot_label()
                    self.v0_slider_elec.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max): self.v0_var_elec.set((v0_max + v0_min) / 2)
                    self._update_v0_label_elec()

                v0 = self.v0_var_elec.get(); potentiel = self.pot_var.get()

            # MODIFIED: Extraire (masse, charge) de self.particles_data
            masse_charge_list_elec = [(p[1], p[2]) for p in self.particles_data]

            self.ax.cla()
            show_uncertainty = self.show_uncertainty_var.get()
            self.status_var.set(f"Calcul déviation électrique {'avec' if show_uncertainty else 'sans'} incertitude...")
            self.root.update_idletasks()

            if show_uncertainty:
                try:
                    incertitudes_dict = { # Assurez-vous que les clés correspondent à deviation.py
                        'v0': float(self.delta_v0_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'theta': float(self.delta_theta_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'h': float(self.delta_h_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'E': float(self.delta_E_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'm': 0.001, # 0.1% fixe
                        'q': 0.001 # MODIFIEE: 0.1% fixe aussi? A verifier dans deviation.py
                    }
                    # Vérifier si deviation.py attend 'E' ou 'potentiel' pour l'incertitude
                    # S'il attend 'potentiel', il faut ajuster la clé ici.
                    deviation.tracer_ensemble_trajectoires_avec_incertitudes(
                        masse_charge_list_elec, vitesse_initiale=v0, incertitudes=incertitudes_dict,
                        potentiel=potentiel, angle_initial=angle_rad, hauteur_initiale=hauteur_initiale,
                        create_plot=False, ax=self.ax
                    )
                    self.status_var.set("Tracé électrique avec incertitudes terminé.")
                except ValueError as e_inc:
                     messagebox.showerror("Erreur Incertitude", f"Valeur d'incertitude invalide:\n{e_inc}", parent=self.root)
                     self.status_var.set("Erreur paramètre incertitude."); return
                except KeyError as e_key:
                    messagebox.showerror("Erreur Code", f"Clé d'incertitude manquante (interne):\n{e_key}", parent=self.root)
                    self.status_var.set("Erreur interne incertitude."); return
            else:
                deviation.tracer_ensemble_trajectoires(
                    masse_charge_list_elec, vitesse_initiale=v0, potentiel=potentiel,
                    angle_initial=angle_rad, hauteur_initiale=hauteur_initiale,
                    create_plot=False, ax=self.ax
                )
                self.status_var.set("Tracé électrique terminé.")

            # MODIFIED: Ajouter les noms aux légendes
            handles, labels = self.ax.get_legend_handles_labels()
            # S'assurer que le nombre de labels correspond aux particules
            num_particles = len(self.particles_data)
            if len(handles) >= num_particles: # Simple vérification
                 new_labels = [f"{self.particles_data[i][0]} ({labels[i]})" for i in range(num_particles)]
                 # Gérer cas incertitude (plus de handles/labels) - Légende simple pour l'instant
                 if show_uncertainty:
                      self.ax.legend(title="Particules (trajectoire moyenne)") # Ou laisser deviation.py gérer
                 else:
                      self.ax.legend(handles[:num_particles], new_labels[:num_particles], title="Particules")
            else:
                 # Laisser la légende par défaut si décalage (p.e. si deviation.py ajoute des choses)
                 self.ax.legend(title="Résultats")


            self.canvas.draw()

        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Électrique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Elec): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur inattendue (Électrique):\n{e}", parent=self.root)
            print(f"Erreur Simulation Électrique: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc() # Imprimer la trace complète pour le débogage
            self.status_var.set("Erreur simulation électrique.")


    # --- Fonctions pour l'onglet Potentiel (ADDED) ---

    def setup_potential_comparison(self):
        """Vérifie la sélection de particule et affiche le panneau de configuration."""
        selected_items = self.particle_tree.selection()

        if not selected_items:
            messagebox.showerror("Erreur", "Veuillez sélectionner une molécule dans la liste ci-dessus.", parent=self.root)
            return
        if len(selected_items) > 1:
            messagebox.showerror("Erreur", "Veuillez sélectionner une SEULE molécule pour la comparaison.", parent=self.root)
            return

        item_id = selected_items[0]
        try:
            index = self.particle_tree.index(item_id)
            # Récupérer les infos depuis self.particles_data en utilisant l'index
            particule_info = self.particles_data[index] # (nom, masse, charge)
            nom = particule_info[0]
            masse = particule_info[1]
            charge = particule_info[2]

            # Stocker les informations nécessaires pour la simulation
            self.selected_potential_particle_info = (index, nom, masse, charge)

            # Mettre à jour le label
            self.selected_particle_label.config(text=f"Particule: {nom} ({masse:.3f} u, {charge:+.2f} e)")

            # Afficher le frame de configuration des potentiels
            self.potential_input_frame.pack(fill=tk.X, pady=10, padx=5, after=self.configure_potential_button)
            self.configure_potential_button.config(state=tk.DISABLED) # Désactiver le bouton config
            self._update_scroll_region_and_bar() # Mettre à jour scroll

        except IndexError:
             messagebox.showerror("Erreur Interne", "Impossible de récupérer les données de la particule sélectionnée.", parent=self.root)
             self.selected_potential_particle_info = None # Réinitialiser
        except Exception as e:
            messagebox.showerror("Erreur Inattendue", f"Une erreur est survenue: {e}", parent=self.root)
            self.selected_potential_particle_info = None # Réinitialiser


    def _reset_potential_comparison_state(self):
        """Cache le frame de configuration et réactive le bouton."""
        self.potential_input_frame.pack_forget()
        self.configure_potential_button.config(state=tk.NORMAL)
        self.selected_potential_particle_info = None
        self.selected_particle_label.config(text="Particule sélectionnée: (aucune)")
        # Effacer aussi le graphe ? Optionnel.
        # self.ax.cla()
        # self.canvas.draw()
        self._update_scroll_region_and_bar() # Mettre à jour scroll
        self.status_var.set("Configuration potentiels réinitialisée.")

    def run_potential_comparison_simulation(self):
        """
        Lance la simulation pour la particule sélectionnée avec deux potentiels différents
        et affiche le delta_x.
        """
        if self.selected_potential_particle_info is None:
            messagebox.showerror("Erreur", "Aucune particule n'est configurée pour la comparaison. Cliquez sur 'Configurer...'.", parent=self.root)
            return

        index, nom, mass_u, charge_e = self.selected_potential_particle_info

        try:
            # --- Lire les paramètres ---
            angle_deg_str = self.angle_var_pot.get().strip().replace(',', '.')
            dist_str = self.dist_var_pot.get().strip().replace(',', '.')
            v0_str = self.v0_var_pot.get().strip().replace(',', '.')
            pot1_str = self.potential1_var.get().strip().replace(',', '.')
            pot2_str = self.potential2_var.get().strip().replace(',', '.')

            if not all([angle_deg_str, dist_str, v0_str, pot1_str, pot2_str]):
                raise ValueError("Tous les paramètres (Angle, Distance, V0, Potentiel 1, Potentiel 2) sont requis.")

            angle_deg = float(angle_deg_str)
            hauteur_distance = float(dist_str)
            v0 = float(v0_str)
            potentiel1 = float(pot1_str)
            potentiel2 = float(pot2_str)

            if hauteur_distance <= 0: raise ValueError("Hauteur/Distance doit être > 0.")
            if not (0 < angle_deg < 90): raise ValueError("Angle doit être > 0° et < 90°.")
            if v0 <= 0: raise ValueError("Vitesse initiale doit être > 0.")
            if potentiel1 == potentiel2: raise ValueError("Les potentiels 1 et 2 doivent être différents pour la comparaison.")

            angle_rad = np.radians(angle_deg)
            hauteur_initiale = hauteur_distance

            # --- Préparer la simulation ---
            self.ax.cla()
            self.status_var.set(f"Calcul comparaison pour {nom}...")
            self.root.update_idletasks()

            # --- Appeler la simulation (Hypothèse sur deviation.py) ---
            # Il faut une fonction qui retourne au moins le point d'impact final
            # Exemple: deviation.calculer_trajectoire_et_impact(masse_u, charge_e, v0, angle_rad, hauteur_init, potentiel)
            # qui retourne (x_coords, y_coords, impact_x) ou juste impact_x si on ne trace pas
            # Ici, on suppose qu'on a besoin des trajectoires pour les tracer.

            # Essayer d'utiliser la fonction existante si elle peut retourner l'impact ou être adaptée
            # Sinon, il faudra créer/modifier deviation.py
            try:
                # Simu 1
                results1 = deviation.calculer_trajectoire( # Renommer si nécessaire
                    [(mass_u, charge_e)], # Liste avec une seule particule
                    vitesse_initiale=v0,
                    potentiel=potentiel1,
                    angle_initial=angle_rad,
                    hauteur_initiale=hauteur_initiale
                )
                # Simu 2
                results2 = deviation.calculer_trajectoire( # Renommer si nécessaire
                    [(mass_u, charge_e)], # Liste avec une seule particule
                    vitesse_initiale=v0,
                    potentiel=potentiel2,
                    angle_initial=angle_rad,
                    hauteur_initiale=hauteur_initiale
                )

                # Extraire les données nécessaires (structure dépend de ce que retourne deviation.calculer_trajectoire)
                # Hypothèse: retourne une liste de dictionnaires, un par particule
                if not results1 or not results2:
                    raise RuntimeError("La simulation n'a pas retourné de résultats.")

                traj1_x = results1[0]['x']
                traj1_y = results1[0]['y']
                impact_x1 = results1[0]['impact_x'] # Clé à adapter selon deviation.py

                traj2_x = results2[0]['x']
                traj2_y = results2[0]['y']
                impact_x2 = results2[0]['impact_x'] # Clé à adapter selon deviation.py

                if impact_x1 is None or impact_x2 is None:
                     raise ValueError("La particule n'a pas atteint la plaque (y=0) pour l'un des potentiels.")

            except AttributeError:
                 messagebox.showerror("Erreur Code", "La fonction 'deviation.calculer_trajectoire' (ou similaire retournant l'impact) semble manquante ou mal nommée dans deviation.py.", parent=self.root)
                 self.status_var.set("Erreur: Fonction de calcul manquante.")
                 return
            except KeyError as e:
                 messagebox.showerror("Erreur Code", f"La clé '{e}' attendue (ex: 'impact_x') n'a pas été trouvée dans les résultats de deviation.py.", parent=self.root)
                 self.status_var.set("Erreur: Format de résultat inattendu.")
                 return
            except Exception as sim_err:
                 raise RuntimeError(f"Erreur lors de l'appel à deviation.calculer_trajectoire: {sim_err}")


            # --- Calculer Delta X ---
            delta_x = abs(impact_x1 - impact_x2)

            # --- Tracer les résultats ---
            label1 = f"Potentiel 1 = {potentiel1:.1f} V (Impact x={impact_x1:.4f} m)"
            label2 = f"Potentiel 2 = {potentiel2:.1f} V (Impact x={impact_x2:.4f} m)"
            self.ax.plot(traj1_x, traj1_y, label=label1, linestyle='-')
            self.ax.plot(traj2_x, traj2_y, label=label2, linestyle='--')

            # Ajouter le Delta X
            delta_x_text = f"Δx = |x₁ - x₂| = {delta_x:.4e} m"
            # Afficher dans le titre ou comme texte sur le graphe
            # self.ax.set_title(f"Comparaison Potentiels pour {nom}\n{delta_x_text}")
            # Ou utiliser ax.text (plus flexible pour le positionnement)
            self.ax.text(0.5, 1.02, delta_x_text, transform=self.ax.transAxes, ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

            self.ax.set_xlabel("Position x (m)")
            self.ax.set_ylabel("Position y (m)")
            self.ax.grid(True)
            self.ax.legend()
            # Ajuster les limites pour bien voir les trajectoires
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            # S'assurer que y=0 est visible
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(min(ymin, -0.01*hauteur_initiale), max(ymax, hauteur_initiale*1.1))


            self.canvas.draw()
            self.status_var.set(f"Comparaison pour {nom} tracée. {delta_x_text}")

        except ValueError as e:
            messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Comparaison Potentiel):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Pot Comp): {e}")
        except RuntimeError as e: # Erreurs venant de la simulation elle-même
            messagebox.showerror("Erreur Simulation", f"{e}", parent=self.root)
            self.status_var.set(f"Erreur simulation (Pot Comp): {e}")
        except Exception as e:
            messagebox.showerror("Erreur Inattendue", f"Une erreur est survenue (Comparaison Potentiel):\n{e}", parent=self.root)
            print(f"Erreur Comparaison Potentiel: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erreur comparaison potentiel.")


    # run_pot_simulation (Supprimé car remplacé par run_potential_comparison_simulation)
    # Les callbacks associés (_on_pot_slider_change_pot, _on_v0_slider_change_elec_pot, etc.) sont aussi supprimés car les widgets correspondants ont disparu.


if __name__ == "__main__":
    root = tk.Tk()
    # Ajouter une police plus grande pour une meilleure lisibilité
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(size=10)
    root.option_add("*Font", default_font)
    app = ParticleApp(root)
    root.mainloop()