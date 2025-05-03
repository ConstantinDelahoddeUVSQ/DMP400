import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font, simpledialog, Listbox
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration des chemins ---
# Note: Assurez-vous que ces chemins sont corrects pour votre structure de projet.
# S'ils ne sont pas corrects, les importations échoueront.
folder = os.path.dirname(os.path.abspath(__file__))
# Ajustez ces chemins si nécessaire pour pointer vers les modules de simulation
path_partie_bleue = os.path.join(folder, "Partie Bleue (accélération)", "Code")
path_partie_verte = os.path.join(folder, "Partie Verte (déviation magnétique)", "Code")
# Ajout du dossier courant pour trouver deviation.py s'il est au même niveau
# sys.path.append(folder) # Décommenter si deviation.py est dans le même dossier
# Correction: Ajouter les chemins des modules *SI* ils existent
if os.path.exists(path_partie_bleue):
    sys.path.append(path_partie_bleue)
else:
    print(f"Avertissement: Chemin non trouvé - {path_partie_bleue}")
if os.path.exists(path_partie_verte):
    sys.path.append(path_partie_verte)
else:
    print(f"Avertissement: Chemin non trouvé - {path_partie_verte}")
# Tentative d'ajout du dossier courant si les autres échouent pour 'deviation'
if not os.path.exists(path_partie_bleue) and not os.path.exists(path_partie_verte):
     if folder not in sys.path:
         sys.path.append(folder)
         print(f"Info: Ajout du dossier courant au path: {folder}")

# --- Importations des modules de simulation ---
try:
    # Assume deviation.py est accessible (soit via les paths ci-dessus, soit dans le dossier courant)
    # Nous aurons besoin d'une fonction pour calculer UNE trajectoire et son impact
    import deviation # type: ignore
    # Importer le module magnétique si nécessaire
    import partie_electroaimant # type: ignore
    print("Modules de simulation importés avec succès.")

    # VÉRIFICATION IMPORTANTE: Assurez-vous que `deviation.py` contient une fonction
    # comme `calculer_trajectoire_et_impact` qui retourne (x_traj, y_traj, impact_x)
    # pour UNE seule particule et UN seul potentiel.
    if not hasattr(deviation, 'calculer_trajectoire_et_impact'):
        print("\n" + "="*50)
        print("ERREUR CRITIQUE: La fonction 'calculer_trajectoire_et_impact'")
        print("n'a pas été trouvée dans le module 'deviation.py'.")
        print("Cette fonction est NÉCESSAIRE pour la nouvelle fonctionnalité")
        print("de l'onglet 'Déviation Électrique (potentiel)'.")
        print("Elle doit accepter: (masse_charge_tuple, vitesse_initiale, potentiel, angle_initial_rad, hauteur_initiale)")
        print("Et retourner: (x_trajectoire, y_trajectoire, x_impact_sur_plaque)")
        print("Veuillez ajouter ou corriger cette fonction dans deviation.py.")
        print("="*50 + "\n")
        # Optionnel: quitter si la fonction manque
        # sys.exit(1)
        # Ou définir une fonction factice pour permettre le lancement (mais la sim ne marchera pas)
        def dummy_sim(mc, v, p, a, h):
            print(f"AVERTISSEMENT: Utilisation de dummy_sim pour {mc}, V={p}")
            return np.array([0, 0.1*abs(p/1000)]), np.array([h, 0]), 0.1*abs(p/1000)
        deviation.calculer_trajectoire_et_impact = dummy_sim


except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Impossible d'importer un ou plusieurs modules de simulation.")
    print(f"Vérifiez l'existence des fichiers .py nécessaires (deviation.py, partie_electroaimant.py).")
    print(f"Chemins vérifiés:")
    print(f"  '{path_partie_bleue}'")
    print(f"  '{path_partie_verte}'")
    print(f"  '{folder}' (dossier courant)")
    print("Assurez-vous que ces chemins sont corrects et que les fichiers existent.")
    # Ne pas quitter immédiatement pour permettre de voir l'interface, mais afficher un message clair
    messagebox.showerror("Erreur d'Importation",
                         f"Impossible d'importer les modules de simulation nécessaires.\n"
                         f"Vérifiez la console pour les détails.\n"
                         f"Erreur: {e}")
    # Optionnellement: sys.exit(1)

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
        self.root.title("Simulateur SIMS - Déviations")
        self.root.geometry("1600x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Style ---
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

        # --- Structure Principale (PanedWindow) ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Panneau de Contrôle Scrollable (Gauche) ---
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

        # Section Onglets Simulations
        self.notebook = ttk.Notebook(control_panel)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        control_panel.rowconfigure(1, weight=1)

        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)
        self.pot_tab = ttk.Frame(self.notebook) # Frame pour l'onglet potentiel

        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')
        self.notebook.add(self.pot_tab, text='Déviation Électrique (potentiel)')

        # Créer les widgets pour chaque onglet
        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)
        # Créer les widgets pour le nouvel onglet potentiel
        self.create_potential_widgets(self.pot_tab) # Appel de la nouvelle fonction

        # --- Panneau Plot (Droite) ---
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

        # --- Barre de Statut ---
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Fonctions de Gestion Scrollbar (INCHANGÉES) ---
    def _bind_mousewheel(self, enter):
        if enter: self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else: self.control_canvas.unbind_all("<MouseWheel>")

    def _update_scroll_region_and_bar(self, event=None):
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        self._update_scrollbar_state()

    def _resize_canvas_content_and_update_bar(self, event=None):
        canvas_width = event.width
        self.control_canvas.itemconfig(self.window_id, width=canvas_width)
        self._update_scrollbar_state()

    def _update_scrollbar_state(self):
        self.root.after(10, self._check_and_set_scrollbar_state)

    def _check_and_set_scrollbar_state(self):
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height: self.scrollbar.pack_forget()
            else: self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except tk.TclError: pass

    def _on_mousewheel(self, event):
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height: return
        except tk.TclError: return

        delta = 0
        if sys.platform == "darwin": # macOS
             delta = event.delta
        else: # Windows/Linux
             delta = event.delta / 120

        if delta < 0: self.control_canvas.yview_scroll(1, "units")
        elif delta > 0: self.control_canvas.yview_scroll(-1, "units")
        return "break"

    # --- Fermeture Propre (INCHANGÉE) ---
    def _on_closing(self):
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le simulateur ?"):
            try: plt.close(self.fig)
            except Exception as e: print(f"Erreur fermeture Matplotlib: {e}")
            try: self.root.destroy()
            except Exception as e: print(f"Erreur destruction Tkinter: {e}")

    # --- Widgets Section Particules (MODIFIÉ pour stocker les noms) ---
    def create_particle_widgets(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill=tk.X)
        input_frame.columnconfigure(1, weight=1); input_frame.columnconfigure(3, weight=1)
        ttk.Label(input_frame, text="Masse (u):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mass_entry = ttk.Entry(input_frame, width=10); self.mass_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.mass_entry.insert(0, "1.0")
        ttk.Label(input_frame, text="Charge (e):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.charge_entry = ttk.Entry(input_frame, width=10); self.charge_entry.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        self.charge_entry.insert(0, "1.0")
        add_btn = ttk.Button(input_frame, text="Ajouter", command=self.add_particle)
        add_btn.grid(row=0, column=4, padx=10, pady=2)

        ttk.Label(input_frame, text="Raccourcis :").grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))
        btns_frame = ttk.Frame(input_frame); btns_frame.grid(row=2, column=0, columnspan=5, pady=5, sticky="ew")
        num_btns = 3; [btns_frame.columnconfigure(i, weight=1) for i in range(num_btns)]
        btn_o2 = ttk.Button(btns_frame, text="O₂⁻", command=lambda: self.ajt_particle_connue(31.998, -1.0))
        btn_o2.grid(row=0, column=0, padx=2, sticky="ew")
        btn_si = ttk.Button(btns_frame, text="Si⁺", command=lambda: self.ajt_particle_connue(28.085, +1.0))
        btn_si.grid(row=0, column=1, padx=2, sticky="ew")
        btn_h = ttk.Button(btns_frame, text="H⁺", command=lambda: self.ajt_particle_connue(1.008, +1.0))
        btn_h.grid(row=0, column=2, padx=2, sticky="ew")

        create_molecule_btn = ttk.Button(parent, text="Construire une Particule...", command=self.ouvrir_fenetre_tp)
        create_molecule_btn.pack(pady=(5, 10), padx=10, fill=tk.X)

        tree_frame = ttk.Frame(parent); tree_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True, ipady=10)
        self.particle_tree = ttk.Treeview(tree_frame, columns=('Name', 'Mass (u)', 'Charge (e)'), show='headings', height=6)
        self.particle_tree.heading('Name', text='Nom'); self.particle_tree.column('Name', width=120, anchor=tk.W)
        self.particle_tree.heading('Mass (u)', text='Masse (u)'); self.particle_tree.column('Mass (u)', width=100, anchor=tk.CENTER)
        self.particle_tree.heading('Charge (e)', text='Charge (e)'); self.particle_tree.column('Charge (e)', width=100, anchor=tk.CENTER)
        scrollbar_tree = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar_tree.set)
        self.particle_tree.bind("<MouseWheel>", lambda e: self._on_mousewheel(e)) # Lier la molette
        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer Sélection", command=self.remove_particle)
        remove_btn.pack(pady=5, padx=10, fill=tk.X)

    # --- Fenêtre Tableau Périodique (INCHANGÉE) ---
    def ouvrir_fenetre_tp(self):
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
                    symbol, mass = element
                    is_lan_act_row = row_idx >= 8
                    btn_style = "LanAct.TButton" if is_lan_act_row else "Element.TButton"
                    btn = ttk.Button(table_frame, text=symbol, width=4, style=btn_style,
                                     command=lambda s=symbol, m=mass: self.construction_de_molecule(s, m))
                    btn.grid(row=row_idx, column=col_idx, padx=1, pady=pady_val, sticky="nsew")

        control_frame = ttk.Frame(self.molecule_fenetre)
        control_frame.pack(pady=10, padx=20, fill=tk.X)
        control_frame.columnconfigure(0, weight=1); control_frame.columnconfigure(1, weight=0); control_frame.columnconfigure(2, weight=1)
        ttk.Label(control_frame, text="Particule construite:", font=('Helvetica', 14, 'bold')).grid(row=0, column=1, sticky="", pady=(0,5))
        display_reset_frame = ttk.Frame(control_frame)
        display_reset_frame.grid(row=1, column=1, sticky="ew", pady=5)
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

    # --- Fonctions de Construction Molécule (INCHANGÉES sauf ajout nom dans submit) ---
    def construction_de_molecule(self, symbol, mass):
        if symbol in self.selected_elts: self.selected_elts[symbol]['count'] += 1
        else: self.selected_elts[symbol] = {'mass': mass, 'count': 1}
        self._update_molecule_display()

    def reset_molecule(self):
        self.selected_elts = {}; self._update_molecule_display()

    def _update_molecule_display(self):
        if not self.selected_elts: self.molecule_display_var.set("(vide)"); return
        sorted_symbols = sorted(self.selected_elts.keys())
        molecule_parts = []
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
        self._add_particle_to_list(mass_u, charge_e, nom) # Passer le nom

    def submit_molecule(self):
        if not self.selected_elts:
            messagebox.showwarning("Aucun Élément", "...", parent=self.molecule_fenetre); return
        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())
            charge_str = self.molecule_charge_var.get().strip().replace(',', '.')
            if not charge_str: raise ValueError("La charge ne peut pas être vide.")
            charge = float(charge_str)
            formula = self.molecule_display_var.get()
            charge_sign = '+' if charge > 0 else '-'
            charge_val = abs(int(charge)) if charge == int(charge) else abs(charge)
            charge_str_fmt = f"({charge_val}{charge_sign})" if charge_val != 1 else f"({charge_sign})"
            nom = f"{formula}{charge_str_fmt}" # Nom formaté

            added = self._add_particle_to_list(total_mass, charge, nom) # Passer le nom
            if added: self.molecule_fenetre.destroy()
        except ValueError as e:
            messagebox.showerror("Erreur de Saisie", f"Erreur: {e}", parent=self.molecule_fenetre)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur inattendue: {e}", parent=self.molecule_fenetre)

    # --- Logique d'ajout et de suppression (MODIFIÉE pour stocker/supprimer nom) ---
    def add_particle(self):
        try:
            mass_u_str = self.mass_entry.get().strip().replace(',', '.')
            charge_e_str = self.charge_entry.get().strip().replace(',', '.')
            if not mass_u_str or not charge_e_str: raise ValueError("Masse et Charge requis.")
            mass_u = float(mass_u_str); charge_e = float(charge_e_str)
            # Créer un nom générique pour l'ajout manuel
            charge_sign = '+' if charge_e > 0 else '-'
            charge_val = abs(int(charge_e)) if charge_e == int(charge_e) else abs(charge_e)
            charge_str_fmt = f"({charge_val}{charge_sign})" if charge_val != 1 else f"({charge_sign})"
            nom = f"Particule {mass_u:.2f}u {charge_str_fmt}"
            self._add_particle_to_list(mass_u, charge_e, nom) # Passer le nom généré
            self.mass_entry.delete(0, tk.END); self.charge_entry.delete(0, tk.END)
        except ValueError as e: messagebox.showerror("Erreur d'Entrée", f"Entrée invalide : {e}")
        except Exception as e: messagebox.showerror("Erreur", f"Erreur inattendue: {e}")

    def _add_particle_to_list(self, mass_u, charge_e, particle_name):
        try:
            if mass_u <= 0: raise ValueError("Masse doit être > 0.")
            if charge_e == 0: raise ValueError("Charge ne peut pas être nulle.")
            current_sign = np.sign(charge_e)
            if self.particles_data:
                existing_sign = np.sign(self.particles_data[0][1])
                if current_sign != existing_sign:
                    raise ValueError("Les particules doivent avoir des charges de même signe.")

            particle_info = (round(mass_u, 5), round(charge_e, 5))
            existing_particles = [(round(p[0], 5), round(p[1], 5)) for p in self.particles_data]

            if particle_info not in existing_particles:
                self.particles_data.append((mass_u, charge_e))
                self.particle_names.append(particle_name) # Ajouter le nom
                self.particle_tree.insert('', tk.END, values=(particle_name, f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                self.status_var.set(f"{particle_name} ajoutée.")
                # Réinitialiser la sélection potentielle si on ajoute une particule
                self._reset_potential_selection()
                return True
            else:
                parent_win = getattr(self, 'molecule_fenetre', self.root)
                messagebox.showwarning("Doublon", f"La particule ({mass_u:.3f}u, {charge_e:+.2f}e) est déjà listée.", parent=parent_win)
                self.status_var.set("Ajout annulé (doublon).")
                return False
        except ValueError as e:
            parent_win = getattr(self, 'molecule_fenetre', self.root)
            if parent_win.winfo_exists(): messagebox.showerror("Erreur Validation", f"{e}", parent=parent_win)
            else: messagebox.showerror("Erreur Validation", f"{e}")
            self.status_var.set(f"Erreur validation: {e}")
            return False
        except Exception as e:
             parent_win = getattr(self, 'molecule_fenetre', self.root)
             if parent_win.winfo_exists(): messagebox.showerror("Erreur Inattendue", f"{e}", parent=parent_win)
             else: messagebox.showerror("Erreur Inattendue", f"{e}")
             self.status_var.set("Erreur interne lors de l'ajout.")
             return False

    def remove_particle(self):
        selected_items = self.particle_tree.selection()
        if not selected_items:
            messagebox.showwarning("Aucune Sélection", "Sélectionnez une particule à supprimer.")
            return
        if len(selected_items) > 1 and not messagebox.askyesno("Confirmation", f"Supprimer les {len(selected_items)} particules ?"):
            return

        indices_to_remove = []
        items_to_remove_tree = []
        for item_id in selected_items:
            try:
                index = self.particle_tree.index(item_id)
                indices_to_remove.append(index)
                items_to_remove_tree.append(item_id)
            except tk.TclError: pass # Déjà supprimé?

        indices_to_remove.sort(reverse=True)
        deleted_count = 0
        for index in indices_to_remove:
            try:
                # Vérifier si la particule supprimée était celle sélectionnée pour potentiel
                if self.selected_potential_particle_index == index:
                     self._reset_potential_selection()

                del self.particles_data[index]
                del self.particle_names[index] # Supprimer le nom aussi
                deleted_count += 1

                # Ajuster l'index sélectionné si une particule avant lui est supprimée
                if self.selected_potential_particle_index is not None and index < self.selected_potential_particle_index:
                    self.selected_potential_particle_index -= 1

            except IndexError: pass

        for item_id in items_to_remove_tree:
            if self.particle_tree.exists(item_id): self.particle_tree.delete(item_id)

        self.status_var.set(f"{deleted_count} particule(s) supprimée(s).")
        # Mettre à jour l'état de l'onglet potentiel si la particule sélectionnée a disparu
        self._update_potential_tab_state()


    # --- Widgets Magnétiques (INCHANGÉS) ---
    def create_magnetic_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        self.dynamic_inputs_frame = ttk.Frame(frame); self.base_inputs_frame = ttk.Frame(frame)
        self.x_detecteur_var = tk.StringVar(value="0.1")
        self.add_labeled_entry(frame, "X détecteur (m):", self.x_detecteur_var).pack(fill=tk.X, pady=3)
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs)
        dynamic_check.pack(anchor=tk.W, pady=5)

        parent_base = self.base_inputs_frame
        self.v0_mag_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)
        self.bz_mag_var = tk.StringVar(value="0.2")
        self.add_labeled_entry(parent_base, "Champ Magnétique (T):", self.bz_mag_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_magnetic_simulation)
        trace_btn_base.pack(pady=15)

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
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_magnetic_simulation)
        apply_limits_btn_dyn.pack(pady=15)
        self.toggle_dynamic_inputs()

    def toggle_dynamic_inputs(self) :
        if self.dynamic_trace_var.get():
            self.base_inputs_frame.pack_forget(); self.dynamic_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else:
            self.dynamic_inputs_frame.pack_forget(); self.base_inputs_frame.pack(fill=tk.X, pady=5, padx=5)

    def _on_bz_slider_change(self, event=None): self._update_bz_label(); self.run_magnetic_simulation(called_by_slider=True)
    def _update_bz_label(self, event=None): self.bz_label_var.set(f"{self.bz_var.get():.3f} T")
    def _on_v0_slider_change(self, event=None): self._update_v0_label(); self.run_magnetic_simulation(called_by_slider=True)
    def _update_v0_label(self, event=None): self.v0_label_var.set(f"{self.v0_var.get():.2e} m/s")

    # --- Widgets Électriques (INCHANGÉS) ---
    def create_electric_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        self.angle_var = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3)
        self.dist_var = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3)

        self.show_uncertainty_var = tk.BooleanVar(value=False)
        self.uncertainty_check = ttk.Checkbutton(frame, text="Afficher Incertitudes", variable=self.show_uncertainty_var, command=self.toggle_uncertainty_inputs)
        self.uncertainty_check.pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.uncertainty_inputs_frame = ttk.LabelFrame(frame, text="Paramètres d'incertitude (%)")
        self.delta_v0_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔV0/V0 (%):", self.delta_v0_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_theta_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "Δθ/θ (% de l'angle):", self.delta_theta_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_h_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "Δh/h (%):", self.delta_h_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_E_percent_var = tk.StringVar(value="1.0"); self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔE/E (%):", self.delta_E_percent_var).pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(self.uncertainty_inputs_frame, text="Note: Δm/m (0.1%) et Δq/q (0.01%) fixes", font=('Segoe UI', 8)).pack(pady=(5,0))

        self.elec_separator = ttk.Separator(frame, orient=tk.HORIZONTAL); self.elec_separator.pack(fill=tk.X, pady=10, padx=5)
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        self.dynamic_elec_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric)
        self.dynamic_elec_check.pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.dynamic_electric_inputs_frame = ttk.Frame(frame); self.base_electric_inputs_frame = ttk.Frame(frame)

        parent_base = self.base_electric_inputs_frame
        self.v0_elec_var = tk.StringVar(value="1e5"); self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)
        self.diff_pot_var = tk.StringVar(value="-5000"); self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_electric_simulation); trace_btn_base.pack(pady=15)

        parent_dyn = self.dynamic_electric_inputs_frame
        self.elec_v0_min_var = tk.StringVar(value="1e4"); self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.elec_v0_min_var).pack(fill=tk.X, pady=3)
        self.elec_v0_max_var = tk.StringVar(value="2e5"); self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.elec_v0_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Vitesse Initiale V0 (m/s):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v0_elec = ttk.Frame(parent_dyn); self.slider_frame_v0_elec.pack(fill=tk.X, pady=(0, 5))
        self.v0_var_elec = tk.DoubleVar(value=1e5)
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=1e4, to=2e5, orient=tk.HORIZONTAL, variable=self.v0_var_elec, command=self._on_v0_slider_change_elec)
        self.v0_slider_elec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var_elec = tk.StringVar(value=f"{self.v0_var_elec.get():.2e} m/s")
        ttk.Label(self.slider_frame_v0_elec, textvariable=self.v0_label_var_elec, width=12).pack(side=tk.LEFT)
        self.diff_pot_min_var = tk.StringVar(value="-10000"); self.add_labeled_entry(parent_dyn, "Potentiel min (V):", self.diff_pot_min_var).pack(fill=tk.X, pady=3)
        self.diff_pot_max_var = tk.StringVar(value="10000"); self.add_labeled_entry(parent_dyn, "Potentiel max (V):", self.diff_pot_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Diff. Potentiel (V):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v = ttk.Frame(parent_dyn); self.slider_frame_v.pack(fill=tk.X, pady=(0, 5))
        self.pot_var = tk.DoubleVar(value=-5000)
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V")
        ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=12).pack(side=tk.LEFT)
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_electric_simulation)
        apply_limits_btn_dyn.pack(pady=15)

        self.toggle_uncertainty_inputs()
        self.toggle_dynamic_electric()

    def toggle_dynamic_electric(self):
        if self.dynamic_elec_var.get():
            self.base_electric_inputs_frame.pack_forget(); self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        else:
            self.dynamic_electric_inputs_frame.pack_forget(); self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, after=self.dynamic_elec_check)
        self.root.after(50, self._update_scroll_region_and_bar)

    def toggle_uncertainty_inputs(self):
        if self.show_uncertainty_var.get(): self.uncertainty_inputs_frame.pack(fill=tk.X, pady=(5,0), padx=5, before=self.elec_separator)
        else: self.uncertainty_inputs_frame.pack_forget()
        self.root.after(50, self._update_scroll_region_and_bar)

    def _on_pot_slider_change(self, event=None): self._update_pot_label(); self.run_electric_simulation(called_by_slider=True)
    def _on_v0_slider_change_elec(self, event=None) : self._update_v0_label_elec(); self.run_electric_simulation(called_by_slider=True)
    def _update_pot_label(self, event=None): self.pot_label_var.set(f"{self.pot_var.get():.1f} V")
    def _update_v0_label_elec(self, event=None): self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e} m/s")


    # --- Widgets Potentiel (NOUVEAU) ---
    def create_potential_widgets(self, parent):
        """
        Crée les widgets pour l'onglet de comparaison de potentiels.
        """
        self.pot_frame = ttk.Frame(parent, padding="10")
        self.pot_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Conteneur pour le bouton de lancement et les contrôles (initialement vide) ---
        self.pot_controls_frame = ttk.Frame(self.pot_frame)
        # NE PAS pack() ce frame ici

        # --- 2. Bouton initial pour lancer la sélection ---
        self.start_pot_sim_button = ttk.Button(self.pot_frame,
                                               text="Sélectionner une Particule...",
                                               command=self.open_particle_selection_window)
        self.start_pot_sim_button.pack(pady=20, padx=10, fill=tk.X)

        # --- 3. Créer les widgets de contrôle (mais ne pas les afficher) ---
        #    Ils seront placés dans self.pot_controls_frame quand une particule est sélectionnée

        # Label pour afficher la particule sélectionnée
        self.selected_particle_label_var = tk.StringVar(value="Aucune particule sélectionnée")
        self.selected_particle_label = ttk.Label(self.pot_controls_frame, textvariable=self.selected_particle_label_var, font=('Helvetica', 10, 'italic'))
        # Ne pas pack() ici

        # Paramètres conservés (Angle, Distance/Hauteur, Vitesse initiale)
        # Utiliser de nouvelles variables pour éviter les conflits avec les autres onglets
        self.angle_pot_var = tk.StringVar(value="30")
        self.angle_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Angle Initial (° vs +y):", self.angle_pot_var)
        # Ne pas pack() ici

        self.dist_pot_var = tk.StringVar(value="0.05")
        self.dist_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Distance/Hauteur (m):", self.dist_pot_var)
        # Ne pas pack() ici

        self.v0_pot_var = tk.StringVar(value="1e5")
        self.v0_pot_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Vitesse Initiale (m/s):", self.v0_pot_var)
        # Ne pas pack() ici

        # Nouvelles entrées pour Potentiel 1 et Potentiel 2
        self.pot1_var = tk.StringVar(value="-5000")
        self.pot1_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Potentiel 1 (V):", self.pot1_var)
        # Ne pas pack() ici

        self.pot2_var = tk.StringVar(value="-6000")
        self.pot2_entry_frame = self.add_labeled_entry(self.pot_controls_frame, "Potentiel 2 (V):", self.pot2_var)
        # Ne pas pack() ici

        # Bouton pour tracer la simulation des deux potentiels
        self.trace_pot_button = ttk.Button(self.pot_controls_frame,
                                            text="Tracer Simulation (2 Potentiels)",
                                            command=self.run_potential_comparison_simulation)
        # Ne pas pack() ici

        # Bouton pour changer de particule
        self.change_particle_button = ttk.Button(self.pot_controls_frame,
                                            text="Changer de Particule",
                                            command=self.open_particle_selection_window) # Réutilise la même fenêtre
        # Ne pas pack() ici

        # Initialiser l'état de l'onglet
        self._update_potential_tab_state()

    def _update_potential_tab_state(self):
        """Affiche soit le bouton de sélection, soit les contrôles de simulation."""
        if self.selected_potential_particle_index is None:
            # Cacher les contrôles et afficher le bouton de sélection
            self.pot_controls_frame.pack_forget()
            self.start_pot_sim_button.pack(pady=20, padx=10, fill=tk.X)
        else:
            # Cacher le bouton de sélection et afficher les contrôles
            self.start_pot_sim_button.pack_forget()
            self.pot_controls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Packer les widgets DANS le pot_controls_frame s'ils ne le sont pas déjà
            if not self.selected_particle_label.winfo_ismapped():
                self.selected_particle_label.pack(pady=(0, 10))
                self.angle_pot_entry_frame.pack(fill=tk.X, pady=3)
                self.dist_pot_entry_frame.pack(fill=tk.X, pady=3)
                self.v0_pot_entry_frame.pack(fill=tk.X, pady=3)
                ttk.Separator(self.pot_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
                self.pot1_entry_frame.pack(fill=tk.X, pady=3)
                self.pot2_entry_frame.pack(fill=tk.X, pady=3)
                self.trace_pot_button.pack(pady=15)
                self.change_particle_button.pack(pady=(0,5))

            # Mettre à jour le label de la particule sélectionnée
            self.selected_particle_label_var.set(f"Particule sélectionnée : {self.selected_potential_particle_name}")

        # S'assurer que la scrollbar est correcte
        self.root.after(50, self._update_scroll_region_and_bar)

    def _reset_potential_selection(self):
        """Réinitialise la sélection de particule pour l'onglet potentiel."""
        self.selected_potential_particle_index = None
        self.selected_potential_particle_name = None
        self._update_potential_tab_state() # Met à jour l'affichage de l'onglet

    def open_particle_selection_window(self):
        """Ouvre une fenêtre pour sélectionner UNE particule pour l'onglet potentiel."""
        if not self.particles_data:
            messagebox.showerror("Pas de Particules",
                                 "Veuillez d'abord ajouter des particules dans la section 'Gestion des Particules'.",
                                 parent=self.root)
            return

        # Créer la fenêtre Toplevel (modale)
        selection_win = tk.Toplevel(self.root)
        selection_win.title("Sélectionner une Particule")
        selection_win.geometry("350x300")
        selection_win.transient(self.root) # Lie la fenêtre à la fenêtre principale
        selection_win.grab_set() # Rend la fenêtre modale (bloque les interactions avec la fenêtre principale)
        selection_win.resizable(False, False)

        ttk.Label(selection_win, text="Choisissez une particule pour la simulation:",
                  font=('Helvetica', 10)).pack(pady=(10, 5))

        # Frame pour la Listbox et Scrollbar
        list_frame = ttk.Frame(selection_win)
        list_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False,
                          font=('Consolas', 10), height=10) # exportselection=False est important

        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Peupler la Listbox avec les noms des particules
        # Stocker l'index original de self.particles_data avec chaque item
        self.listbox_to_data_index = {}
        for i, name in enumerate(self.particle_names):
            # Afficher nom, masse, charge pour aider à la sélection
            mass, charge = self.particles_data[i]
            display_text = f"{name} (m={mass:.2f}u, q={charge:+.1f}e)"
            listbox.insert(tk.END, display_text)
            self.listbox_to_data_index[listbox.size() - 1] = i # Map listbox index to data index

        # Pré-sélectionner l'élément courant s'il existe
        if self.selected_potential_particle_index is not None:
            # Trouver l'index de la listbox correspondant à l'index des données
            try:
                listbox_idx = list(self.listbox_to_data_index.values()).index(self.selected_potential_particle_index)
                listbox.selection_set(listbox_idx)
                listbox.activate(listbox_idx)
                listbox.see(listbox_idx)
            except ValueError:
                pass # L'index n'est plus valide (devrait pas arriver si _reset fonctionne)


        # Fonction pour confirmer la sélection
        def confirm():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Aucune Sélection", "Veuillez sélectionner une particule.", parent=selection_win)
                return

            listbox_index = selected_indices[0]
            original_data_index = self.listbox_to_data_index[listbox_index]

            # Stocker l'index et le nom de la particule sélectionnée
            self.selected_potential_particle_index = original_data_index
            self.selected_potential_particle_name = self.particle_names[original_data_index]

            selection_win.destroy()
            self._update_potential_tab_state() # Mettre à jour l'onglet principal
            self.status_var.set(f"Particule '{self.selected_potential_particle_name}' sélectionnée pour comparaison.")

        # Fonction pour annuler
        def cancel():
            selection_win.destroy()

        # Frame pour les boutons
        button_frame = ttk.Frame(selection_win)
        button_frame.pack(pady=(5, 10))

        select_btn = ttk.Button(button_frame, text="Sélectionner", command=confirm)
        select_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = ttk.Button(button_frame, text="Annuler", command=cancel)
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # Centrer la fenêtre
        selection_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (selection_win.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (selection_win.winfo_height() // 2)
        selection_win.geometry(f"+{x}+{y}")

        selection_win.wait_window() # Attend que la fenêtre soit fermée

    # --- Helper (INCHANGÉ) ---
    def add_labeled_entry(self, parent, label_text, string_var):
        entry_frame = ttk.Frame(parent)
        entry_frame.columnconfigure(0, weight=0); entry_frame.columnconfigure(1, weight=1)
        label = ttk.Label(entry_frame, text=label_text, anchor="w")
        label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.grid(row=0, column=1, sticky="ew", pady=2)
        return entry_frame

    # --- Exécution des Simulations ---

    # Simulation Magnétique (INCHANGÉE)
    def run_magnetic_simulation(self, called_by_slider=False):
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler."); self.ax.cla(); self.canvas.draw(); return
        try:
            x_detecteur = float(self.x_detecteur_var.get().strip().replace(',', '.'))
            if x_detecteur <= 0: raise ValueError("X détecteur doit être positif.")
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
            # Assurez-vous que partie_electroaimant existe et a la fonction tracer_ensemble_trajectoires
            if hasattr(partie_electroaimant, 'tracer_ensemble_trajectoires'):
                 partie_electroaimant.tracer_ensemble_trajectoires(
                     self.particles_data, v0, bz, x_detecteur, create_plot=False, ax=self.ax, labels_particules=self.particle_names # Passer les noms
                 )
            else:
                 messagebox.showerror("Erreur Module", "Fonction 'tracer_ensemble_trajectoires' non trouvée dans 'partie_electroaimant'.")
                 self.status_var.set("Erreur module magnétique.")
                 return

            self.ax.relim(); self.ax.autoscale_view(True, True, True); self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")
        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Magnétique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Mag): {e}")
        except NameError as e:
             if 'partie_electroaimant' in str(e):
                  messagebox.showerror("Erreur Module", "Le module 'partie_electroaimant' n'a pas pu être importé.")
             else:
                  messagebox.showerror("Erreur", f"Erreur Nom (Magnétique): {e}")
             self.status_var.set("Erreur module/nom (Mag).")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur (Magnétique):\n{type(e).__name__}: {e}", parent=self.root)
            self.status_var.set("Erreur simulation magnétique.")

    # Simulation Électrique Standard (INCHANGÉE)
    def run_electric_simulation(self, called_by_slider=False):
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Ajoutez au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules."); self.ax.cla(); self.canvas.draw(); return
        try:
            angle_deg = float(self.angle_var.get().strip().replace(',', '.'))
            hauteur_distance = float(self.dist_var.get().strip().replace(',', '.'))
            if hauteur_distance <= 0 : raise ValueError("Hauteur/Distance > 0.")
            if not (0 < angle_deg < 90): raise ValueError("0° < Angle < 90°.")
            angle_rad = np.radians(angle_deg)
            hauteur_initiale = hauteur_distance

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
            self.ax.cla()
            show_uncertainty = self.show_uncertainty_var.get()
            self.status_var.set(f"Calcul déviation électrique {'avec' if show_uncertainty else 'sans'} incertitude...")
            self.root.update_idletasks()

            # Assurez-vous que deviation existe et a les fonctions nécessaires
            if not hasattr(deviation, 'tracer_ensemble_trajectoires'):
                 messagebox.showerror("Erreur Module", "Fonction 'tracer_ensemble_trajectoires' non trouvée dans 'deviation'.")
                 self.status_var.set("Erreur module déviation.")
                 return

            if show_uncertainty:
                if not hasattr(deviation, 'tracer_ensemble_trajectoires_avec_incertitudes'):
                    messagebox.showerror("Erreur Module", "Fonction 'tracer_ensemble_trajectoires_avec_incertitudes' non trouvée dans 'deviation'.")
                    self.status_var.set("Erreur module déviation (incertitudes).")
                    return
                try:
                    incertitudes_dict = {
                        'v0': float(self.delta_v0_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'theta': float(self.delta_theta_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'h': float(self.delta_h_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'E': float(self.delta_E_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'm': 0.001, 'q': 0.0001
                    }
                    deviation.tracer_ensemble_trajectoires_avec_incertitudes(
                        masse_charge_list, vitesse_initiale=v0, incertitudes=incertitudes_dict,
                        potentiel=potentiel, angle_initial=angle_rad, hauteur_initiale=hauteur_initiale,
                        create_plot=False, ax=self.ax, labels_particules=self.particle_names # Passer les noms
                    )
                    self.status_var.set("Tracé électrique avec incertitudes terminé.")
                except ValueError as e_inc:
                     messagebox.showerror("Erreur Incertitude", f"Valeur invalide:\n{e_inc}", parent=self.root)
                     self.status_var.set("Erreur paramètre incertitude."); return
                except KeyError as e_key:
                    messagebox.showerror("Erreur Code", f"Clé manquante (interne):\n{e_key}", parent=self.root)
                    self.status_var.set("Erreur interne incertitude."); return
            else:
                deviation.tracer_ensemble_trajectoires(
                    masse_charge_list, vitesse_initiale=v0, potentiel=potentiel,
                    angle_initial=angle_rad, hauteur_initiale=hauteur_initiale,
                    create_plot=False, ax=self.ax, labels_particules=self.particle_names # Passer les noms
                )
                self.status_var.set("Tracé électrique terminé.")

            self.canvas.draw()

        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Électrique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Elec): {e}")
        except NameError as e:
             if 'deviation' in str(e):
                  messagebox.showerror("Erreur Module", "Le module 'deviation' n'a pas pu être importé.")
             else:
                  messagebox.showerror("Erreur", f"Erreur Nom (Électrique): {e}")
             self.status_var.set("Erreur module/nom (Elec).")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur (Électrique):\n{type(e).__name__}: {e}", parent=self.root)
            self.status_var.set("Erreur simulation électrique.")


    # Simulation Comparaison Potentiels (NOUVEAU)
    def run_potential_comparison_simulation(self):
        """
        Lance la simulation pour la particule sélectionnée avec deux potentiels différents
        et affiche le delta(x) d'impact.
        """
        if self.selected_potential_particle_index is None:
            messagebox.showerror("Erreur", "Aucune particule n'est sélectionnée pour cette simulation.", parent=self.root)
            return

        # Vérifier si la fonction nécessaire existe dans deviation
        if not hasattr(deviation, 'calculer_trajectoire_et_impact'):
             messagebox.showerror("Erreur Module",
                                  "La fonction 'calculer_trajectoire_et_impact' est requise dans deviation.py\n"
                                  "Impossible de lancer cette simulation.", parent=self.root)
             self.status_var.set("Erreur: Fonction 'calculer_trajectoire_et_impact' manquante.")
             return

        try:
            # Récupérer les paramètres communs (angle, hauteur, v0) de CET onglet
            angle_deg = float(self.angle_pot_var.get().strip().replace(',', '.'))
            hauteur_distance = float(self.dist_pot_var.get().strip().replace(',', '.'))
            v0 = float(self.v0_pot_var.get().strip().replace(',', '.'))

            # Récupérer les deux potentiels
            potentiel1 = float(self.pot1_var.get().strip().replace(',', '.'))
            potentiel2 = float(self.pot2_var.get().strip().replace(',', '.'))

            # Validation simple
            if hauteur_distance <= 0: raise ValueError("Hauteur/Distance doit être > 0.")
            if not (0 < angle_deg < 90): raise ValueError("Angle doit être > 0° et < 90°.")
            if v0 <= 0: raise ValueError("Vitesse initiale doit être > 0.")
            # Optionnel: vérifier si potentiel1 == potentiel2 ?

            angle_rad = np.radians(angle_deg)
            hauteur_initiale = hauteur_distance

            # Récupérer les données de la particule sélectionnée
            particle_data = self.particles_data[self.selected_potential_particle_index]
            particle_name = self.selected_potential_particle_name

            self.status_var.set(f"Calcul pour {particle_name} avec V={potentiel1}V et V={potentiel2}V...")
            self.ax.cla() # Nettoyer le graphe précédent
            self.root.update_idletasks()

            # --- Simulation 1 ---
            x_traj1, y_traj1, impact_x1 = deviation.calculer_trajectoire_et_impact(
                particle_data, v0, potentiel1, angle_rad, hauteur_initiale
            )

            # --- Simulation 2 ---
            x_traj2, y_traj2, impact_x2 = deviation.calculer_trajectoire_et_impact(
                particle_data, v0, potentiel2, angle_rad, hauteur_initiale
            )

            # --- Affichage ---
            # Tracer les deux trajectoires
            line1, = self.ax.plot(x_traj1, y_traj1, label=f'Potentiel 1 = {potentiel1:.1f} V')
            line2, = self.ax.plot(x_traj2, y_traj2, label=f'Potentiel 2 = {potentiel2:.1f} V')

            # Calculer et afficher le delta(x)
            delta_x_str = "N/A"
            delta_x = None
            if impact_x1 is not None and impact_x2 is not None:
                # S'assurer que ce sont bien des nombres avant de soustraire
                if isinstance(impact_x1, (int, float)) and isinstance(impact_x2, (int, float)):
                    if not (np.isnan(impact_x1) or np.isnan(impact_x2)):
                        delta_x = abs(impact_x1 - impact_x2)
                        delta_x_str = f"{delta_x:.3e} m"
                    else:
                         delta_x_str = "Impact NaN"
                else:
                     delta_x_str = "Impact non numérique"

            # Ajouter la plaque (y=0)
            xlim = self.ax.get_xlim()
            xmin = 0 # Assurer xmin = 0
            xmax = max(xlim[1], impact_x1 if impact_x1 is not None and not np.isnan(impact_x1) else 0,
                                impact_x2 if impact_x2 is not None and not np.isnan(impact_x2) else 0) * 1.1
            self.ax.plot([xmin, xmax], [0, 0], 'k--', label='Plaque détectrice (y=0)') # Ligne pour la plaque
            self.ax.set_xlim(xmin, xmax)


            # Mise en forme du graphe
            self.ax.set_xlabel("Position x (m)")
            self.ax.set_ylabel("Position y (m)")
            self.ax.set_title(f"Comparaison de potentiels pour {particle_name}\n V0={v0:.1e} m/s, Angle={angle_deg}°, h0={hauteur_initiale}m")
            self.ax.grid(True, linestyle=':')

            # Créer une légende personnalisée incluant le delta(x)
            # On récupère les handles et labels existants
            handles, labels = self.ax.get_legend_handles_labels()

            # Ajouter une ligne factice pour le texte delta_x si on veut le mettre dans la légende
            # Ou l'ajouter comme titre de légende
            legend_title = f"Particule: {particle_name}\nΔx (impact) = {delta_x_str}"
            self.ax.legend(handles=handles, labels=labels, title=legend_title, loc='best')

            # Adapter les limites y si nécessaire (le module deviation le fait peut-être déjà bien)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True) # Ajuster les axes
            # S'assurer que l'axe y commence au moins à 0 s'il y a des impacts
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(min(ymin, -0.001), ymax) # Laisser un petit espace sous 0


            self.canvas.draw()
            self.status_var.set(f"Simulation terminée. Δx = {delta_x_str}")

        except ValueError as e:
            messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Potentiel):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Potentiel): {e}")
        except NameError as e:
             if 'deviation' in str(e):
                  messagebox.showerror("Erreur Module", "Le module 'deviation' n'a pas pu être importé.")
             else:
                  messagebox.showerror("Erreur", f"Erreur Nom (Potentiel): {e}")
             self.status_var.set("Erreur module/nom (Potentiel).")
        except Exception as e:
            messagebox.showerror("Erreur Simulation", f"Erreur inattendue (Potentiel):\n{type(e).__name__}: {e}", parent=self.root)
            print(f"Erreur Simulation Potentiel: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc() # Imprimer la trace complète dans la console pour le débogage
            self.status_var.set("Erreur simulation potentiel.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()