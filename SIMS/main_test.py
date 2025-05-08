# --- START OF FILE main_final.py ---

import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font, Listbox
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration des chemins (revue pour robustesse) ---
folder = os.path.dirname(os.path.abspath(__file__))
# Créer les chemins absolus
path_partie_bleue = os.path.abspath(os.path.join(folder, "Partie Bleue (accélération)", "Code"))
path_partie_verte = os.path.abspath(os.path.join(folder, "Partie Verte (déviation magnétique)", "Code"))

# Ajouter les chemins au sys.path s'ils existent et ne sont pas déjà présents
paths_to_add = [path_partie_bleue, path_partie_verte]
for pth in paths_to_add:
    if os.path.isdir(pth) and pth not in sys.path:
        sys.path.append(pth)
    # else: # Décommenter pour debug
    #     print(f"Info: Chemin ignoré ou inexistant - {pth}")

# Ajouter le dossier courant si besoin (pour deviation_test.py)
if folder not in sys.path:
    sys.path.append(folder)

# --- Importations des modules de simulation ---
try:
    # Utiliser le nom final du module
    import deviation_test as deviation # type : ignore
    import partie_electroaimant_test as partie_electroaimant# type : ignore
    print("Modules de simulation importés.")
except ImportError as e:
    print(f"ERREUR FATALE d'importation: {e}")
    print("Impossible d'importer les modules de simulation nécessaires (deviation_test, partie_electroaimant_test).")
    print("Vérifiez que les fichiers .py existent et que les chemins sont corrects.")
    print("Chemins ajoutés au sys.path:")
    for p in sys.path: print(f"  - {p}")
    messagebox.showerror("Erreur d'Importation", f"Modules non trouvés: {e}\nVérifiez la console.")
    sys.exit(1)
except Exception as e_other:
     print(f"ERREUR FATALE lors de l'import: {type(e_other).__name__}: {e_other}")
     messagebox.showerror("Erreur Inconnue", f"Erreur lors de l'import: {e_other}")
     sys.exit(1)


# --- Vérification fonction critique ---
# S'assurer que la fonction nécessaire pour le nouvel onglet est bien présente
if not hasattr(deviation, 'calculer_trajectoire_et_impact'):
    print("\n" + "="*50)
    print("ERREUR CRITIQUE: La fonction 'calculer_trajectoire_et_impact'")
    print("n'est pas définie dans le module 'deviation_test.py'.")
    print("Cette fonction est INDISPENSABLE pour l'onglet 'Potentiel'.")
    print("Veuillez l'ajouter ou corriger son nom dans deviation_test.py.")
    print("="*50 + "\n")
    messagebox.showerror("Erreur Module",
                         "Fonction 'calculer_trajectoire_et_impact' manquante dans deviation_test.py.\n"
                         "L'onglet 'Potentiel' ne fonctionnera pas.")
    # On ne quitte pas, mais l'onglet sera inutilisable


# --- Classe principale de l'application ---
class ParticleApp:
    def __init__(self, root):
        # ... (Initialisation root, titre, protocol, style - INCHANGÉ) ...
        self.root = root
        self.root.title("Simulateur SIMS - Déviations vFinal")
        self.root.geometry("1600x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        style = ttk.Style()
        try: style.theme_use('vista')
        except tk.TclError:
            try: style.theme_use('aqua')
            except tk.TclError: style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat")
        style.configure("TLabelframe.Label", font=('Helvetica', 13, 'bold'))
        style.configure("TLabel", padding=2)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        style.configure("Element.TButton", padding=2, font=('Segoe UI', 9))
        style.configure("LanAct.TButton", padding=2, font=('Segoe UI', 9), background="#e8f4ea")

        # --- Données ---
        self.particles_data = [] # Liste de tuples (masse_u: float, charge_e: float)
        self.particle_names = [] # Liste parallèle pour stocker les noms/formules

        # --- Données spécifiques à l'onglet Potentiel ---
        self.selected_potential_particle_index = None # Index dans self.particles_data
        self.selected_potential_particle_name = None

        # --- Structure Principale et Panneau de Contrôle Scrollable (INCHANGÉ) ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)
        container_frame = ttk.Frame(main_paned_window, width=450)
        container_frame.pack_propagate(False)
        main_paned_window.add(container_frame, weight=0)
        self.control_canvas = tk.Canvas(container_frame)
        self.scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.control_canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self._update_scroll_region_and_bar(e))
        self.window_id = self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.bind("<Configure>", self._resize_canvas_content_and_update_bar)
        self.control_canvas.bind("<Enter>", lambda e: self._bind_mousewheel(True))
        self.control_canvas.bind("<Leave>", lambda e: self._bind_mousewheel(False))
        control_panel = self.scrollable_frame

        # --- Widgets dans le Panneau de Contrôle ---
        particle_frame = ttk.LabelFrame(control_panel, text="Gestion des Particules")
        particle_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_panel.columnconfigure(0, weight=1)
        self.create_particle_widgets(particle_frame)

        self.notebook = ttk.Notebook(control_panel)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        control_panel.rowconfigure(1, weight=1)

        # Créer les frames pour les onglets
        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)
        self.pot_tab = ttk.Frame(self.notebook) # Frame pour le nouvel onglet

        # Ajouter les onglets au Notebook
        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')
        self.notebook.add(self.pot_tab, text='Comparaison Potentiels') # Nom mis à jour

        # Créer les widgets pour chaque onglet
        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)
        self.create_potential_widgets(self.pot_tab) # Appel de la nouvelle fonction

        # --- Panneau Plot (Droite) (INCHANGÉ) ---
        plot_panel = ttk.Frame(main_paned_window)
        main_paned_window.add(plot_panel, weight=1)
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Barre de Statut (INCHANGÉ) ---
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Fonctions Scrollbar, Fermeture, Tableau Périodique (INCHANGÉES) ---
    # ... (Copier les méthodes _bind_mousewheel, _update_scroll_region_and_bar,
    #      _resize_canvas_content_and_update_bar, _update_scrollbar_state,
    #      _check_and_set_scrollbar_state, _on_mousewheel, _on_closing,
    #      ouvrir_fenetre_tp, construction_de_molecule, reset_molecule,
    #      _update_molecule_display, ajt_particle_connue, submit_molecule
    #      depuis main_v3.py ici, elles sont correctes) ...
    # --- Fonctions de Gestion Scrollbar ---
    def _bind_mousewheel(self, enter):
        """Lie ou délie les événements de molette pour le canvas."""
        if enter:
            self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel) # Windows/Mac
            self.control_canvas.bind_all("<Button-4>", self._on_mousewheel) # Linux Scroll Up
            self.control_canvas.bind_all("<Button-5>", self._on_mousewheel) # Linux Scroll Down
        else:
            self.control_canvas.unbind_all("<MouseWheel>")
            self.control_canvas.unbind_all("<Button-4>")
            self.control_canvas.unbind_all("<Button-5>")

    def _update_scroll_region_and_bar(self, event=None):
        """Met à jour la scrollregion ET l'état de la scrollbar."""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        self._update_scrollbar_state()

    def _resize_canvas_content_and_update_bar(self, event=None):
        """Redimensionne le contenu interne et met à jour l'état de la scrollbar."""
        canvas_width = event.width
        self.control_canvas.itemconfig(self.window_id, width=canvas_width)
        self._update_scrollbar_state()

    def _update_scrollbar_state(self):
        """Active ou désactive la scrollbar si le contenu dépasse."""
        # Utiliser after pour laisser le temps à Tk de calculer les tailles
        self.root.after(10, self._check_and_set_scrollbar_state)

    def _check_and_set_scrollbar_state(self):
        """Vérifie et active/désactive la scrollbar."""
        try:
            # Comparer la hauteur requise par le contenu au hauteur réelle du canvas
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            # print(f"Canvas H: {canvas_height}, Content H: {content_height}") # Utile pour debug
            if content_height <= canvas_height:
                # Cacher la scrollbar si pas nécessaire
                if self.scrollbar.winfo_ismapped():
                    self.scrollbar.pack_forget()
            else:
                # Afficher la scrollbar si nécessaire
                if not self.scrollbar.winfo_ismapped():
                    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except tk.TclError:
            # Ignorer l'erreur si la fenêtre est en cours de fermeture
            pass

    def _on_mousewheel(self, event):
        """Gère le défilement par molette si scroll nécessaire."""
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height:
                return # Pas besoin de scroller
        except tk.TclError:
             return # Erreur si fenêtre fermée

        # Calculer le facteur de défilement
        delta = 0
        if event.num == 5 or event.delta < 0: # Scroll down (ou delta négatif sur certains systèmes)
            delta = 1
        elif event.num == 4 or event.delta > 0: # Scroll up (ou delta positif)
            delta = -1

        if delta != 0:
            self.control_canvas.yview_scroll(delta, "units")
            # Mise à jour immédiate pour fluidité (peut être coûteux si contenu complexe)
            # self.root.update_idletasks()
        return "break" # Empêche d'autres widgets de recevoir l'événement


    # --- Fermeture Propre ---
    def _on_closing(self):
        """Gère la fermeture de la fenêtre."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le simulateur ?"):
            try:
                plt.close(self.fig) # Fermer la figure matplotlib
            except Exception as e:
                print(f"Erreur lors de la fermeture de Matplotlib: {e}")
            try:
                self.root.destroy() # Fermer la fenêtre Tkinter
            except Exception as e:
                print(f"Erreur lors de la destruction de Tkinter: {e}")

    # --- Widgets Section Particules (avec gestion nom) ---
    def create_particle_widgets(self, parent):
        """Crée les widgets pour la gestion des particules."""
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill=tk.X)
        input_frame.columnconfigure(1, weight=1); input_frame.columnconfigure(3, weight=1)

        ttk.Label(input_frame, text="Masse (u):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mass_entry = ttk.Entry(input_frame, width=10); self.mass_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.mass_entry.insert(0, "1.0")
        ttk.Label(input_frame, text="Charge (e):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.charge_entry = ttk.Entry(input_frame, width=10); self.charge_entry.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        self.charge_entry.insert(0, "1.0")
        add_btn = ttk.Button(input_frame, text="Ajouter", command=self.add_particle); add_btn.grid(row=0, column=4, padx=10, pady=2)

        ttk.Label(input_frame, text="Raccourcis :").grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))
        btns_frame = ttk.Frame(input_frame); btns_frame.grid(row=2, column=0, columnspan=5, pady=5, sticky="ew")
        num_btns = 3; [btns_frame.columnconfigure(i, weight=1) for i in range(num_btns)]
        btn_o2 = ttk.Button(btns_frame, text="O₂⁻", command=lambda: self.ajt_particle_connue(31.998, -1.0)); btn_o2.grid(row=0, column=0, padx=2, sticky="ew")
        btn_si = ttk.Button(btns_frame, text="Si⁺", command=lambda: self.ajt_particle_connue(28.085, +1.0)); btn_si.grid(row=0, column=1, padx=2, sticky="ew")
        btn_h = ttk.Button(btns_frame, text="H⁺", command=lambda: self.ajt_particle_connue(1.008, +1.0)); btn_h.grid(row=0, column=2, padx=2, sticky="ew")

        create_molecule_btn = ttk.Button(parent, text="Construire une Particule...", command=self.ouvrir_fenetre_tp)
        create_molecule_btn.pack(pady=(5, 10), padx=10, fill=tk.X)

        tree_frame = ttk.Frame(parent); tree_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True, ipady=10)
        self.particle_tree = ttk.Treeview(tree_frame, columns=('Name', 'Mass (u)', 'Charge (e)'), show='headings', height=6)
        self.particle_tree.heading('Name', text='Nom'); self.particle_tree.column('Name', width=100, anchor=tk.CENTER)
        self.particle_tree.heading('Mass (u)', text='Masse (u)'); self.particle_tree.column('Mass (u)', width=100, anchor=tk.CENTER)
        self.particle_tree.heading('Charge (e)', text='Charge (e)'); self.particle_tree.column('Charge (e)', width=100, anchor=tk.CENTER)
        scrollbar_tree = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar_tree.set)
        self.particle_tree.bind("<MouseWheel>", lambda e: self._on_mousewheel(e))
        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer Sélection", command=self.remove_particle)
        remove_btn.pack(pady=5, padx=10, fill=tk.X)


    # --- Fenêtre Tableau Périodique ---
    def ouvrir_fenetre_tp(self):
        """Ouvre une fenêtre modale pour construire une molécule/ion."""
        if hasattr(self, 'molecule_fenetre') and self.molecule_fenetre.winfo_exists():
            self.molecule_fenetre.lift(); return

        self.molecule_fenetre = tk.Toplevel(self.root)
        self.molecule_fenetre.title("Construire une Particule")
        self.molecule_fenetre.geometry("1100x650")
        self.molecule_fenetre.grab_set()
        self.molecule_fenetre.transient(self.root)
        self.selected_elts = {}

        periodic_layout = [
            [('H', 1.008), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ('He', 4.0026)],
            [('Li', 6.94), ('Be', 9.0122), None, None, None, None, None, None, None, None, None, None, ('B', 10.81), ('C', 12.011), ('N', 14.007), ('O', 15.999), ('F', 18.998), ('Ne', 20.180)],
            [('Na', 22.990), ('Mg', 24.305), None, None, None, None, None, None, None, None, None, None, ('Al', 26.982), ('Si', 28.085), ('P', 30.974), ('S', 32.06), ('Cl', 35.45), ('Ar', 39.948)],
            [('K', 39.098), ('Ca', 40.078), ('Sc', 44.956), ('Ti', 47.867), ('V', 50.942), ('Cr', 51.996), ('Mn', 54.938), ('Fe', 55.845), ('Co', 58.933), ('Ni', 58.693), ('Cu', 63.546), ('Zn', 65.38), ('Ga', 69.723), ('Ge', 72.630), ('As', 74.922), ('Se', 78.971), ('Br', 79.904), ('Kr', 83.798)],
            [('Rb', 85.468), ('Sr', 87.62), ('Y', 88.906), ('Zr', 91.224), ('Nb', 92.906), ('Mo', 95.95), ('Tc', 98.0), ('Ru', 101.07), ('Rh', 102.91), ('Pd', 106.42), ('Ag', 107.87), ('Cd', 112.41), ('In', 114.82), ('Sn', 118.71), ('Sb', 121.76), ('Te', 127.60), ('I', 126.90), ('Xe', 131.29)],
            [('Cs', 132.91), ('Ba', 137.33), ('La', 138.91), ('Hf', 178.49), ('Ta', 180.95), ('W', 183.84), ('Re', 186.21), ('Os', 190.23), ('Ir', 192.22), ('Pt', 195.08), ('Au', 196.97), ('Hg', 200.59), ('Tl', 204.38), ('Pb', 207.2), ('Bi', 208.98), ('Po', 209.0), ('At', 210.0), ('Rn', 222.0)],
            [('Fr', 223.0), ('Ra', 226.0), ('Ac', 227.0), ('Rf', 267.0), ('Db', 270.0), ('Sg', 271.0), ('Bh', 270.0), ('Hs', 277.0), ('Mt', 278.0), ('Ds', 281.0), ('Rg', 282.0), ('Cn', 285.0), ('Nh', 286.0), ('Fl', 289.0), ('Mc', 290.0), ('Lv', 293.0), ('Ts', 294.0), ('Og', 294.0)],
            [],
            [None, None, None, ('Ce', 140.12), ('Pr', 140.91), ('Nd', 144.24), ('Pm', 145.0), ('Sm', 150.36), ('Eu', 151.96), ('Gd', 157.25), ('Tb', 158.93), ('Dy', 162.50), ('Ho', 164.93), ('Er', 167.26), ('Tm', 168.93), ('Yb', 173.05), ('Lu', 174.97), None],
            [None, None, None, ('Th', 232.04), ('Pa', 231.04), ('U', 238.03), ('Np', 237.0), ('Pu', 244.0), ('Am', 243.0), ('Cm', 247.0), ('Bk', 247.0), ('Cf', 251.0), ('Es', 252.0), ('Fm', 257.0), ('Md', 258.0), ('No', 259.0), ('Lr', 262.0), None]
        ]
        table_frame = ttk.Frame(self.molecule_fenetre); table_frame.pack(pady=10, padx=10)
        for row_idx, row in enumerate(periodic_layout):
            pady_val = 5 if row_idx == 8 else 2
            for col_idx, element in enumerate(row):
                if element:
                    symbol, mass = element; is_lan_act = row_idx >= 8
                    btn_style = "LanAct.TButton" if is_lan_act else "Element.TButton"
                    btn = ttk.Button(table_frame, text=symbol, width=4, style=btn_style,
                                     command=lambda s=symbol, m=mass: self.construction_de_molecule(s, m))
                    btn.grid(row=row_idx, column=col_idx, padx=1, pady=pady_val, sticky="nsew")

        control_frame = ttk.Frame(self.molecule_fenetre); control_frame.pack(pady=10, padx=20, fill=tk.X)
        control_frame.columnconfigure(0, weight=1); control_frame.columnconfigure(1, weight=0); control_frame.columnconfigure(2, weight=1)
        ttk.Label(control_frame, text="Particule construite:", font=('Helvetica', 14, 'bold')).grid(row=0, column=1, sticky="", pady=(0,5))
        display_reset_frame = ttk.Frame(control_frame); display_reset_frame.grid(row=1, column=1, sticky="ew", pady=5)
        display_reset_frame.columnconfigure(0, weight=1)
        self.molecule_display_var = tk.StringVar(value="(vide)")
        display_label = ttk.Label(display_reset_frame, textvariable=self.molecule_display_var, relief=tk.SUNKEN, padding=5, anchor=tk.W, font=('Consolas', 10))
        display_label.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))
        reset_btn = ttk.Button(display_reset_frame, text="Effacer", command=self.reset_molecule)
        reset_btn.grid(row=0, column=1, sticky=tk.E)
        charge_frame = ttk.Frame(control_frame); charge_frame.grid(row=2, column=1, sticky="", pady=5)
        charge_label = ttk.Label(charge_frame, text="Charge (e):"); charge_label.pack(side=tk.LEFT)
        self.molecule_charge_var = tk.StringVar(value="1")
        charge_entry = ttk.Entry(charge_frame, textvariable=self.molecule_charge_var, width=8); charge_entry.pack(side=tk.LEFT, padx=5)
        submit_btn = ttk.Button(control_frame, text="Ajouter cette Particule à la liste", command=self.submit_molecule)
        submit_btn.grid(row=3, column=1, sticky="", pady=10)

    def construction_de_molecule(self, symbol, mass):
        if symbol in self.selected_elts: self.selected_elts[symbol]['count'] += 1
        else: self.selected_elts[symbol] = {'mass': mass, 'count': 1}
        self._update_molecule_display()

    def reset_molecule(self):
        self.selected_elts = {}; self._update_molecule_display()

    def _update_molecule_display(self):
        if not self.selected_elts: self.molecule_display_var.set("(vide)"); return
        sorted_symbols = sorted(self.selected_elts.keys()); molecule_parts = []
        for symbol in sorted_symbols:
            data = self.selected_elts[symbol]; count = data['count']; part = symbol
            if count > 1:
                subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                part += str(count).translate(subscript_map)
            molecule_parts.append(part)
        self.molecule_display_var.set("".join(molecule_parts))

    def ajt_particle_connue(self, mass_u, charge_e):
        symbole = ""
        if mass_u == 31.998: symbole = "O₂"
        elif mass_u == 28.085: symbole = "Si"
        elif mass_u == 1.008: symbole = "H"
        else: symbole = f"{mass_u:.3f}u"
        charge_sign = '+' if charge_e > 0 else '-'
        charge_val = abs(int(charge_e)) if charge_e == int(charge_e) else abs(charge_e)
        charge_str = f"({charge_val}{charge_sign})" if charge_val != 1 else f"({charge_sign})"
        nom = f"{symbole}{charge_str}"
        self._add_particle_to_list(mass_u, charge_e, nom)

    def submit_molecule(self):
        if not self.selected_elts: messagebox.showwarning("Aucun Élément", "...", parent=self.molecule_fenetre); return
        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())
            charge_str = self.molecule_charge_var.get().strip().replace(',', '.')
            if not charge_str: raise ValueError("Charge vide.")
            charge = float(charge_str)
            formula = self.molecule_display_var.get()
            charge_sign = '+' if charge > 0 else '-'
            charge_val = abs(int(charge)) if charge == int(charge) else abs(charge)
            charge_str_fmt = f"({charge_val}{charge_sign})" if charge_val != 1 else f"({charge_sign})"
            nom = f"{formula}{charge_str_fmt}"
            added = self._add_particle_to_list(total_mass, charge, nom)
            if added: self.molecule_fenetre.destroy()
        except ValueError as e: messagebox.showerror("Erreur Saisie", f"Erreur: {e}", parent=self.molecule_fenetre)
        except Exception as e: messagebox.showerror("Erreur", f"Erreur inattendue: {e}", parent=self.molecule_fenetre)

    # --- Logique d'ajout/suppression (gère nom) ---
    def add_particle(self):
        try:
            mass_u = float(self.mass_entry.get().strip().replace(',', '.'))
            charge_e = float(self.charge_entry.get().strip().replace(',', '.'))
            charge_sign = '+' if charge_e > 0 else '-'
            charge_val = abs(int(charge_e)) if charge_e == int(charge_e) else abs(charge_e)
            charge_str_fmt = f"({charge_val}{charge_sign})" if charge_val != 1 else f"({charge_sign})"
            nom = f"Particule {mass_u:.2f}u {charge_str_fmt}"
            self._add_particle_to_list(mass_u, charge_e, nom)
            self.mass_entry.delete(0, tk.END); self.charge_entry.delete(0, tk.END)
        except ValueError as e: messagebox.showerror("Erreur d'Entrée", f"Entrée invalide : {e}")
        except Exception as e: messagebox.showerror("Erreur", f"Erreur inattendue: {e}")

    def _add_particle_to_list(self, mass_u, charge_e, particle_name):
        try:
            if mass_u <= 0: raise ValueError("Masse > 0.")
            if charge_e == 0: raise ValueError("Charge != 0.")
            current_sign = np.sign(charge_e)
            if self.particles_data:
                existing_sign = np.sign(self.particles_data[0][1])
                if current_sign != existing_sign: raise ValueError("Charges de même signe requises.")

            particle_info = (round(mass_u, 5), round(charge_e, 5))
            existing_particles = [(round(p[0], 5), round(p[1], 5)) for p in self.particles_data]

            if particle_info not in existing_particles:
                self.particles_data.append((mass_u, charge_e))
                self.particle_names.append(particle_name)
                self.particle_tree.insert('', tk.END, values=(particle_name, f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                self.status_var.set(f"{particle_name} ajoutée.")
                self._reset_potential_selection() # Réinitialiser au cas où
                return True
            else:
                parent_win = getattr(self, 'molecule_fenetre', self.root)
                messagebox.showwarning("Doublon", f"Particule déjà listée.", parent=parent_win)
                return False
        except ValueError as e:
            parent_win = getattr(self, 'molecule_fenetre', self.root)
            if parent_win.winfo_exists(): messagebox.showerror("Erreur Validation", f"{e}", parent=parent_win)
            else: messagebox.showerror("Erreur Validation", f"{e}")
            return False
        except Exception as e:
             parent_win = getattr(self, 'molecule_fenetre', self.root)
             if parent_win.winfo_exists(): messagebox.showerror("Erreur Inattendue", f"{e}", parent=parent_win)
             else: messagebox.showerror("Erreur Inattendue", f"{e}")
             return False

    def remove_particle(self):
        selected_items = self.particle_tree.selection()
        if not selected_items: return
        if len(selected_items) > 1 and not messagebox.askyesno("Confirmation", f"Supprimer {len(selected_items)} particules ?"): return

        indices_to_remove = []
        items_to_remove_tree = []
        for item_id in selected_items:
            try: indices_to_remove.append(self.particle_tree.index(item_id)); items_to_remove_tree.append(item_id)
            except tk.TclError: pass

        indices_to_remove.sort(reverse=True)
        deleted_count = 0
        for index in indices_to_remove:
            try:
                if self.selected_potential_particle_index == index: self._reset_potential_selection()
                del self.particles_data[index]; del self.particle_names[index]; deleted_count += 1
                if self.selected_potential_particle_index is not None and index < self.selected_potential_particle_index:
                    self.selected_potential_particle_index -= 1
            except IndexError: pass

        for item_id in items_to_remove_tree:
            if self.particle_tree.exists(item_id): self.particle_tree.delete(item_id)

        self.status_var.set(f"{deleted_count} particule(s) supprimée(s).")
        self._update_potential_tab_state() # Mettre à jour l'affichage si besoin

    # --- Widgets Onglet Magnétique (INCHANGÉ) ---
    def create_magnetic_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        self.dynamic_inputs_frame = ttk.Frame(frame); self.base_inputs_frame = ttk.Frame(frame)
        self.x_detecteur_var = tk.StringVar(value="0.1")
        self.add_labeled_entry(frame, "X détecteur (m):", self.x_detecteur_var).pack(fill=tk.X, pady=3)
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs); dynamic_check.pack(anchor=tk.W, pady=5)

        parent_base = self.base_inputs_frame
        self.v0_mag_var = tk.StringVar(value="1e6"); self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)
        self.bz_mag_var = tk.StringVar(value="0.2"); self.add_labeled_entry(parent_base, "Champ Magnétique (T):", self.bz_mag_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_magnetic_simulation); trace_btn_base.pack(pady=15)

        parent_dyn = self.dynamic_inputs_frame
        self.bz_min_var = tk.StringVar(value="0.01"); self.add_labeled_entry(parent_dyn, "Bz min (T):", self.bz_min_var).pack(fill=tk.X, pady=3)
        self.bz_max_var = tk.StringVar(value="0.5"); self.add_labeled_entry(parent_dyn, "Bz max (T):", self.bz_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_bz = ttk.Frame(parent_dyn); self.slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=0.2)
        self.bz_slider = ttk.Scale(self.slider_frame_bz, from_=0.01, to=0.5, orient=tk.HORIZONTAL, variable=self.bz_var, command=self._on_bz_slider_change)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(self.slider_frame_bz, textvariable=self.bz_label_var, width=10).pack(side=tk.LEFT)
        self.v0_min_var = tk.StringVar(value="1e5"); self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.v0_min_var).pack(fill=tk.X, pady=3)
        self.v0_max_var = tk.StringVar(value="1e6"); self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.v0_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Vitesse Initiale (m/s):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_v0 = ttk.Frame(parent_dyn); self.slider_frame_v0.pack(fill=tk.X, pady=(0,5))
        self.v0_var = tk.DoubleVar(value=5.5e5)
        self.v0_slider = ttk.Scale(self.slider_frame_v0, from_=1e5, to=1e6, orient=tk.HORIZONTAL, variable=self.v0_var, command=self._on_v0_slider_change)
        self.v0_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var = tk.StringVar(value=f"{self.v0_var.get():.2e} m/s")
        ttk.Label(self.slider_frame_v0, textvariable=self.v0_label_var, width=12).pack(side=tk.LEFT)
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_magnetic_simulation); apply_limits_btn_dyn.pack(pady=15)
        self.toggle_dynamic_inputs()

    def toggle_dynamic_inputs(self) :
        if self.dynamic_trace_var.get(): self.base_inputs_frame.pack_forget(); self.dynamic_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else: self.dynamic_inputs_frame.pack_forget(); self.base_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        self.root.after(50, self._update_scroll_region_and_bar)

    def _on_bz_slider_change(self, event=None): self._update_bz_label(); self.run_magnetic_simulation(called_by_slider=True)
    def _update_bz_label(self, event=None): self.bz_label_var.set(f"{self.bz_var.get():.3f}\u00A0T") # Espace insécable
    def _on_v0_slider_change(self, event=None): self._update_v0_label(); self.run_magnetic_simulation(called_by_slider=True)
    def _update_v0_label(self, event=None): self.v0_label_var.set(f"{self.v0_var.get():.2e}\u00A0m/s")

    # --- Widgets Onglet Électrique Standard (INCHANGÉ) ---
    def create_electric_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        self.angle_var = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3)
        self.dist_var = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3)

        self.show_uncertainty_var = tk.BooleanVar(value=False)
        self.uncertainty_check = ttk.Checkbutton(frame, text="Afficher Incertitudes", variable=self.show_uncertainty_var, command=self.toggle_uncertainty_inputs); self.uncertainty_check.pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.uncertainty_inputs_frame = ttk.LabelFrame(frame, text="Paramètres d'incertitude relative (%)")
        self.delta_v0_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔV₀/V₀ (%):", self.delta_v0_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_theta_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "Δθ/θ (%):", self.delta_theta_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_h_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "Δh/h (%):", self.delta_h_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_E_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔE/E (%):", self.delta_E_percent_var).pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(self.uncertainty_inputs_frame, text="Note: Δm/m (0.1%), Δq/q (0.01%) fixes", font=('Segoe UI', 8)).pack(pady=(5,0))

        self.elec_separator = ttk.Separator(frame, orient=tk.HORIZONTAL); self.elec_separator.pack(fill=tk.X, pady=10, padx=5)
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        self.dynamic_elec_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric); self.dynamic_elec_check.pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.dynamic_electric_inputs_frame = ttk.Frame(frame); self.base_electric_inputs_frame = ttk.Frame(frame)
        parent_base = self.base_electric_inputs_frame
        self.v0_elec_var = tk.StringVar(value="1e5"); self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)
        self.diff_pot_var = tk.StringVar(value="-5000"); self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_electric_simulation); trace_btn_base.pack(pady=15)
        parent_dyn = self.dynamic_electric_inputs_frame
        self.elec_v0_min_var = tk.StringVar(value="1e4"); self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.elec_v0_min_var).pack(fill=tk.X, pady=3)
        self.elec_v0_max_var = tk.StringVar(value="2e5"); self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.elec_v0_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Vitesse Initiale V₀ (m/s):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v0_elec = ttk.Frame(parent_dyn); self.slider_frame_v0_elec.pack(fill=tk.X, pady=(0, 5))
        self.v0_var_elec = tk.DoubleVar(value=1e5)
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=1e4, to=2e5, orient=tk.HORIZONTAL, variable=self.v0_var_elec, command=self._on_v0_slider_change_elec); self.v0_slider_elec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var_elec = tk.StringVar(value=f"{self.v0_var_elec.get():.2e} m/s"); ttk.Label(self.slider_frame_v0_elec, textvariable=self.v0_label_var_elec, width=12).pack(side=tk.LEFT)
        self.diff_pot_min_var = tk.StringVar(value="-10000"); self.add_labeled_entry(parent_dyn, "Potentiel min (V):", self.diff_pot_min_var).pack(fill=tk.X, pady=3)
        self.diff_pot_max_var = tk.StringVar(value="10000"); self.add_labeled_entry(parent_dyn, "Potentiel max (V):", self.diff_pot_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Diff. Potentiel (V):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v = ttk.Frame(parent_dyn); self.slider_frame_v.pack(fill=tk.X, pady=(0, 5))
        self.pot_var = tk.DoubleVar(value=-5000)
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._on_pot_slider_change); self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V"); ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=12).pack(side=tk.LEFT)
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_electric_simulation); apply_limits_btn_dyn.pack(pady=15)
        self.toggle_uncertainty_inputs(); self.toggle_dynamic_electric()

    def toggle_dynamic_electric(self):
        if self.dynamic_elec_var.get(): self.base_electric_inputs_frame.pack_forget(); self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        else: self.dynamic_electric_inputs_frame.pack_forget(); self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        self.root.after(50, self._update_scroll_region_and_bar)

    def toggle_uncertainty_inputs(self):
        if self.show_uncertainty_var.get(): self.uncertainty_inputs_frame.pack(fill=tk.X, pady=(5,0), padx=5, before=self.elec_separator)
        else: self.uncertainty_inputs_frame.pack_forget()
        self.root.after(50, self._update_scroll_region_and_bar)

    def _on_pot_slider_change(self, event=None): self._update_pot_label(); self.run_electric_simulation(called_by_slider=True)
    def _on_v0_slider_change_elec(self, event=None) : self._update_v0_label_elec(); self.run_electric_simulation(called_by_slider=True)
    def _update_pot_label(self, event=None): self.pot_label_var.set(f"{self.pot_var.get():.1f}\u00A0V") # Espace insécable
    def _update_v0_label_elec(self, event=None): self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e}\u00A0m/s")


    # --- Widgets Onglet Potentiel (NOUVEAU) ---
    def create_potential_widgets(self, parent):
        """Crée les widgets pour l'onglet de comparaison de potentiels."""
        self.pot_frame = ttk.Frame(parent, padding="10")
        self.pot_frame.pack(fill=tk.BOTH, expand=True)

        # Frame pour les contrôles (caché initialement)
        self.pot_controls_frame = ttk.Frame(self.pot_frame)
        # Bouton pour démarrer (visible initialement)
        self.start_pot_sim_button = ttk.Button(self.pot_frame, text="Sélectionner Particule pour Comparaison...", command=self.open_particle_selection_window)
        self.start_pot_sim_button.pack(pady=20, padx=10, fill=tk.X)

        # --- Widgets dans le frame de contrôles ---
        self.selected_particle_label_var = tk.StringVar(value="Aucune particule sélectionnée")
        self.selected_particle_label = ttk.Label(self.pot_controls_frame, textvariable=self.selected_particle_label_var, font=('Helvetica', 10, 'italic', 'bold'), anchor='center')

        self.angle_pot_var = tk.StringVar(value="30")
        self.angle_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Angle Initial (° vs +y):", self.angle_pot_var)

        self.dist_pot_var = tk.StringVar(value="0.05")
        self.dist_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Distance/Hauteur (m):", self.dist_pot_var)

        self.v0_pot_var = tk.StringVar(value="1e5")
        self.v0_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Vitesse Initiale (m/s):", self.v0_pot_var)

        # Potentiels à comparer
        self.pot1_var = tk.StringVar(value="0") # Potentiel référence souvent 0V
        self.pot1_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Potentiel Référence (V):", self.pot1_var)

        self.pot2_var = tk.StringVar(value="-5000") # Potentiel de test
        self.pot2_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Potentiel Test (V):", self.pot2_var)

        self.trace_pot_button = ttk.Button(self.pot_controls_frame, text="Tracer Comparaison & Calculer Δx", command=self.run_potential_comparison_simulation)
        self.change_particle_button = ttk.Button(self.pot_controls_frame, text="Changer Particule", command=self.open_particle_selection_window)

        # Affichage initial
        self._update_potential_tab_state()

    def _update_potential_tab_state(self):
        """Affiche/cache widgets de l'onglet Potentiel."""
        if self.selected_potential_particle_index is None:
            self.pot_controls_frame.pack_forget()
            self.start_pot_sim_button.pack(pady=20, padx=10, fill=tk.X)
        else:
            self.start_pot_sim_button.pack_forget()
            self.pot_controls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            if not self.selected_particle_label.winfo_ismapped():
                # Packer dans l'ordre souhaité
                self.selected_particle_label.pack(pady=(5, 10))
                self.angle_pot_entry_frame.pack(fill=tk.X, pady=3)
                self.dist_pot_entry_frame.pack(fill=tk.X, pady=3)
                self.v0_pot_entry_frame.pack(fill=tk.X, pady=3)
                ttk.Separator(self.pot_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
                self.pot1_entry_frame.pack(fill=tk.X, pady=3)
                self.pot2_entry_frame.pack(fill=tk.X, pady=3)
                self.trace_pot_button.pack(pady=15)
                self.change_particle_button.pack(pady=(0,10))
            self.selected_particle_label_var.set(f"Particule : {self.selected_potential_particle_name}")
        self.root.after(50, self._update_scroll_region_and_bar)

    def _reset_potential_selection(self):
        """Réinitialise la sélection pour l'onglet potentiel."""
        self.selected_potential_particle_index = None
        self.selected_potential_particle_name = None
        self._update_potential_tab_state()

    def open_particle_selection_window(self):
        """Ouvre la fenêtre modale pour sélectionner UNE particule."""
        if not self.particles_data:
            messagebox.showerror("Pas de Particules", "Ajoutez des particules d'abord.", parent=self.root)
            return

        selection_win = tk.Toplevel(self.root)
        selection_win.title("Sélectionner Particule")
        selection_win.geometry("400x300") # Taille ajustée
        selection_win.transient(self.root); selection_win.grab_set(); selection_win.resizable(False, True)

        ttk.Label(selection_win, text="Choisissez une particule:", font=('Helvetica', 10)).pack(pady=(10, 5))
        list_frame = ttk.Frame(selection_win); list_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False, font=('Consolas', 10), height=10)
        scrollbar.config(command=listbox.yview); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listbox.bind("<MouseWheel>", lambda e: listbox.yview_scroll(int(-1*(e.delta/120)), "units")) # Scroll listbox

        self.listbox_to_data_index = {}
        for i, name in enumerate(self.particle_names):
            mass, charge = self.particles_data[i]
            display_text = f"{name} (m={mass:.2f}u, q={charge:+.1f}e)"
            listbox.insert(tk.END, display_text)
            self.listbox_to_data_index[listbox.size() - 1] = i

        if self.selected_potential_particle_index is not None:
            try:
                listbox_idx = list(self.listbox_to_data_index.values()).index(self.selected_potential_particle_index)
                listbox.selection_set(listbox_idx); listbox.activate(listbox_idx); listbox.see(listbox_idx)
            except ValueError: pass

        def confirm():
            selected_indices = listbox.curselection()
            if not selected_indices: messagebox.showwarning("Aucune Sélection", "Sélectionnez une particule.", parent=selection_win); return
            original_data_index = self.listbox_to_data_index[selected_indices[0]]
            self.selected_potential_particle_index = original_data_index
            self.selected_potential_particle_name = self.particle_names[original_data_index]
            selection_win.destroy()
            self._update_potential_tab_state()
            self.status_var.set(f"Particule '{self.selected_potential_particle_name}' sélectionnée.")

        def cancel(): selection_win.destroy()

        button_frame = ttk.Frame(selection_win); button_frame.pack(pady=(5, 10))
        select_btn = ttk.Button(button_frame, text="Sélectionner", command=confirm); select_btn.pack(side=tk.LEFT, padx=10)
        cancel_btn = ttk.Button(button_frame, text="Annuler", command=cancel); cancel_btn.pack(side=tk.LEFT, padx=10)

        # Centrage fenêtre
        selection_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width()//2) - (selection_win.winfo_width()//2)
        y = self.root.winfo_y() + (self.root.winfo_height()//2) - (selection_win.winfo_height()//2)
        selection_win.geometry(f"+{x}+{y}")
        selection_win.wait_window()


    # --- Helper (INCHANGÉ) ---
    def add_labeled_entry(self, parent, label_text, string_var):
        entry_frame = ttk.Frame(parent)
        entry_frame.columnconfigure(0, weight=0); entry_frame.columnconfigure(1, weight=1)
        label = ttk.Label(entry_frame, text=label_text, anchor="w"); label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        entry = ttk.Entry(entry_frame, textvariable=string_var); entry.grid(row=0, column=1, sticky="ew", pady=2)
        return entry_frame

    # --- Exécution des Simulations ---

    # Simulation Magnétique (INCHANGÉE - mais passe les noms)
    def run_magnetic_simulation(self, called_by_slider=False):
        # ... (Code identique à la version précédente, s'assurer qu'il passe self.particle_names au backend si besoin) ...
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Ajoutez des particules.", parent=self.root)
            self.status_var.set("Ajoutez des particules."); self.ax.cla(); self.canvas.draw(); return
        try:
            x_detecteur = float(self.x_detecteur_var.get().strip().replace(',', '.'))
            if x_detecteur <= 0: raise ValueError("X détecteur > 0.")
            if not self.dynamic_trace_var.get():
                v0 = float(self.v0_mag_var.get().strip().replace(',', '.'))
                bz = float(self.bz_mag_var.get().strip().replace(',', '.'))
                if v0 <= 0: raise ValueError("V0 > 0.")
                if bz == 0: raise ValueError("Bz != 0.")
            else:
                if not called_by_slider:
                    bz_min = float(self.bz_min_var.get().strip().replace(',', '.'))
                    bz_max = float(self.bz_max_var.get().strip().replace(',', '.'))
                    v0_min = float(self.v0_min_var.get().strip().replace(',', '.'))
                    v0_max = float(self.v0_max_var.get().strip().replace(',', '.'))
                    if v0_min <= 0: raise ValueError("V0 min > 0.")
                    if v0_min >= v0_max: raise ValueError("V0 max > V0 min.")
                    if bz_min >= bz_max: raise ValueError("Bz max > Bz min.")
                    current_bz = self.bz_var.get(); current_v0 = self.v0_var.get()
                    self.bz_slider.config(from_=bz_min, to=bz_max)
                    if not (bz_min <= current_bz <= bz_max): self.bz_var.set((bz_max + bz_min) / 2)
                    self._update_bz_label()
                    self.v0_slider.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max): self.v0_var.set((v0_max + v0_min) / 2)
                    self._update_v0_label()
                v0 = self.v0_var.get(); bz = self.bz_var.get()
                if abs(bz) < 1e-15: raise ValueError("Bz trop proche de zéro.")

            self.ax.cla(); self.status_var.set("Calcul déviation magnétique..."); self.root.update_idletasks()
            # Appel Backend - passe les data (m,c) et les noms séparément
            partie_electroaimant.tracer_ensemble_trajectoires(
                self.particles_data, # Le backend devrait extraire (m,c)
                v0, bz, x_detecteur, create_plot=False, ax=self.ax,
                labels_particules=self.particle_names # Passer les noms
            )
            self.ax.relim(); self.ax.autoscale_view(True, True, True); self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")
        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Inv. (Mag): {e}", parent=self.root)
            self.status_var.set(f"Erreur param (Mag): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur", f"Erreur (Mag):\n{type(e).__name__}: {e}", parent=self.root)
            self.status_var.set("Erreur sim. mag.")


    # Simulation Électrique Standard (INCHANGÉE - mais passe les noms)
    def run_electric_simulation(self, called_by_slider=False):
        # ... (Code identique à la version précédente, s'assurer qu'il passe self.particle_names au backend) ...
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Ajoutez des particules.", parent=self.root)
            self.status_var.set("Ajoutez des particules."); self.ax.cla(); self.canvas.draw(); return
        try:
            angle_deg = float(self.angle_var.get().strip().replace(',', '.'))
            hauteur_distance = float(self.dist_var.get().strip().replace(',', '.'))
            if hauteur_distance <= 0 : raise ValueError("Hauteur/Distance > 0.")
            if not (0 < angle_deg < 90): raise ValueError("0° < Angle < 90°.")
            angle_rad = np.radians(angle_deg); hauteur_initiale = hauteur_distance

            if not self.dynamic_elec_var.get():
                v0 = float(self.v0_elec_var.get().strip().replace(',', '.'))
                potentiel = float(self.diff_pot_var.get().strip().replace(',', '.'))
                if v0 <= 0 : raise ValueError("V0 > 0.")
            else:
                if not called_by_slider:
                    pot_min = float(self.diff_pot_min_var.get().strip().replace(',', '.'))
                    pot_max = float(self.diff_pot_max_var.get().strip().replace(',', '.'))
                    v0_min = float(self.elec_v0_min_var.get().strip().replace(',', '.'))
                    v0_max = float(self.elec_v0_max_var.get().strip().replace(',', '.'))
                    if v0_min <= 0: raise ValueError("V0 min > 0.")
                    if v0_min >= v0_max: raise ValueError("V0 max > V0 min.")
                    if pot_min >= pot_max: raise ValueError("Potentiel max > Potentiel min.")
                    current_pot = self.pot_var.get(); current_v0 = self.v0_var_elec.get()
                    self.pot_slider.config(from_=pot_min, to=pot_max)
                    if not (pot_min <= current_pot <= pot_max): self.pot_var.set((pot_max + pot_min) / 2)
                    self._update_pot_label()
                    self.v0_slider_elec.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max): self.v0_var_elec.set((v0_max + v0_min) / 2)
                    self._update_v0_label_elec()
                v0 = self.v0_var_elec.get(); potentiel = self.pot_var.get()

            masse_charge_list = self.particles_data
            self.ax.cla(); show_uncertainty = self.show_uncertainty_var.get()
            self.status_var.set(f"Calcul déviation électrique {'avec' if show_uncertainty else 'sans'} incertitude..."); self.root.update_idletasks()

            if show_uncertainty:
                try:
                    incertitudes_dict = {'v0': float(self.delta_v0_percent_var.get().strip().replace(',', '.'))/100,'theta': float(self.delta_theta_percent_var.get().strip().replace(',', '.'))/100,'h': float(self.delta_h_percent_var.get().strip().replace(',', '.'))/100,'E': float(self.delta_E_percent_var.get().strip().replace(',', '.'))/100,'m': 0.001,'q': 0.0001}
                    deviation.tracer_ensemble_trajectoires_avec_incertitudes(masse_charge_list, vitesse_initiale=v0, incertitudes=incertitudes_dict, potentiel=potentiel, angle_initial=angle_rad, hauteur_initiale=hauteur_initiale, create_plot=False, ax=self.ax, labels_particules=self.particle_names)
                    self.status_var.set("Tracé électrique avec incertitudes terminé.")
                except ValueError as e: messagebox.showerror("Erreur Incertitude", f"Inv: {e}", parent=self.root); self.status_var.set("Erreur param incertitude."); return
                except KeyError as e: messagebox.showerror("Erreur Code", f"Clé: {e}", parent=self.root); self.status_var.set("Erreur interne incertitude."); return
            else:
                deviation.tracer_ensemble_trajectoires(masse_charge_list, vitesse_initiale=v0, potentiel=potentiel, angle_initial=angle_rad, hauteur_initiale=hauteur_initiale, create_plot=False, ax=self.ax, labels_particules=self.particle_names)
                self.status_var.set("Tracé électrique terminé.")

            self.canvas.draw()
        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Inv. (Elec): {e}", parent=self.root)
            self.status_var.set(f"Erreur param (Elec): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur", f"Erreur (Elec):\n{type(e).__name__}: {e}", parent=self.root)
            self.status_var.set("Erreur sim. elec.")


    # Simulation Comparaison Potentiels (NOUVEAU)
    def run_potential_comparison_simulation(self):
        """Lance la simulation pour la particule sélectionnée avec deux potentiels."""
        if self.selected_potential_particle_index is None:
            messagebox.showerror("Erreur", "Sélectionnez une particule pour cette simulation.", parent=self.root)
            return

        if not hasattr(deviation, 'calculer_trajectoire_et_impact'):
             messagebox.showerror("Erreur Module", "Fonction 'calculer_trajectoire_et_impact' requise.", parent=self.root)
             return

        try:
            # Lire params de l'onglet Potentiel
            angle_deg = float(self.angle_pot_var.get().strip().replace(',', '.'))
            hauteur_distance = float(self.dist_pot_var.get().strip().replace(',', '.'))
            v0 = float(self.v0_pot_var.get().strip().replace(',', '.'))
            potentiel1 = float(self.pot1_var.get().strip().replace(',', '.'))
            potentiel2 = float(self.pot2_var.get().strip().replace(',', '.'))

            # Validation
            if hauteur_distance <= 0: raise ValueError("Hauteur/Distance > 0.")
            if not (0 < angle_deg < 90): raise ValueError("0° < Angle < 90°.")
            if v0 <= 0: raise ValueError("V0 > 0.")

            angle_rad = np.radians(angle_deg); hauteur_initiale = hauteur_distance
            particle_data = self.particles_data[self.selected_potential_particle_index]
            particle_name = self.selected_potential_particle_name

            self.status_var.set(f"Calcul pour {particle_name} (V1={potentiel1}V, V2={potentiel2}V)...")
            self.ax.cla(); self.root.update_idletasks()

            # Appels Backend
            xt1, yt1, imp1 = deviation.calculer_trajectoire_et_impact(particle_data, v0, potentiel1, angle_rad, hauteur_initiale)
            xt2, yt2, imp2 = deviation.calculer_trajectoire_et_impact(particle_data, v0, potentiel2, angle_rad, hauteur_initiale)

            # Affichage
            line1, = self.ax.plot(xt1, yt1, label=f'V₁ = {potentiel1:.1f} V')
            line2, = self.ax.plot(xt2, yt2, label=f'V₂ = {potentiel2:.1f} V')

            delta_x_str = "N/A"; delta_x = None
            imp1_valid = isinstance(imp1, (int, float)) and not np.isnan(imp1)
            imp2_valid = isinstance(imp2, (int, float)) and not np.isnan(imp2)

            if imp1_valid and imp2_valid:
                delta_x = abs(imp1 - imp2)
                delta_x_str = f"{delta_x:.3e} m"
            elif imp1_valid:
                 delta_x_str = f"Seul V₁ impacte ({imp1:.3e} m)"
            elif imp2_valid:
                 delta_x_str = f"Seul V₂ impacte ({imp2:.3e} m)"
            else:
                 delta_x_str = "Aucun impact > 0"


            # Plaque
            xmax_plot = max([0] + ([imp1] if imp1_valid else []) + ([imp2] if imp2_valid else []) + [hauteur_initiale]) * 1.15
            self.ax.plot([0, xmax_plot], [0, 0], 'k-', linewidth=2, label='Plaque (y=0)')
            self.ax.set_xlim(left=0, right=xmax_plot)

            # Annoter les points d'impact si > 0
            if imp1_valid: self.ax.plot(imp1, 0, 'o', color=line1.get_color(), markersize=5)
            if imp2_valid: self.ax.plot(imp2, 0, 'x', color=line2.get_color(), markersize=6)

            self.ax.set_xlabel("Position x (m)"); self.ax.set_ylabel("Position y (m)")
            self.ax.set_title(f"Comparaison Potentiels pour {particle_name}\nΔx = {delta_x_str}")
            self.ax.grid(True, linestyle=':')
            self.ax.legend(loc='best'); self.ax.relim(); self.ax.autoscale_view(True, True, True)
            ymin, ymax = self.ax.get_ylim(); self.ax.set_ylim(min(ymin, -0.01*hauteur_initiale), ymax) # Espace sous y=0

            self.canvas.draw()
            self.status_var.set(f"Comparaison tracée. Δx = {delta_x_str}")

        except ValueError as e: messagebox.showerror("Erreur Paramètre", f"Inv. (Potentiel): {e}", parent=self.root); self.status_var.set(f"Erreur param (Pot): {e}")
        except NameError as e: messagebox.showerror("Erreur Module", f"'{e}' non trouvé. Module manquant?", parent=self.root); self.status_var.set("Erreur module/nom (Pot).")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur (Pot):\n{type(e).__name__}: {e}", parent=self.root)
            import traceback; traceback.print_exc()
            self.status_var.set("Erreur simulation potentiel.")

# --- Point d'entrée ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()

# --- END OF FILE main_final.py ---