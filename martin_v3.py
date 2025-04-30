import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.abspath(__file__))
path_partie_bleue = os.path.join(folder, "SIMS", "Partie Bleue (accélération)", "Code")
path_partie_verte = os.path.join(folder, "SIMS", "Partie Verte (déviation magnétique)", "Code")
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
    print(f"  '{path_partie_bleue}'")
    print(f"  '{path_partie_verte}'")
    print("Assurez-vous que ces dossiers sont corrects et contiennent les fichiers __init__.py si nécessaire.")
    sys.exit(1)

# --- Fonctions Utilitaires pour Incertitude (Électrique) ---

def champ_electrique_v2(distance: float, différence_potentiel: float) -> float:
    """Calcule E = V/d."""
    if abs(distance) < 1e-15: # Éviter division par zéro
        # Si V est aussi nul, E=0. Si V non nul, E est infini (cas non physique ici).
        if abs(différence_potentiel) < 1e-15:
            return 0.0
        else:
            # On pourrait lever une erreur, mais pour la simu, retourner une valeur très grande
            # ou simplement 0 si on considère que d=0 n'est pas permis.
            # Levons une erreur pour forcer une distance valide.
             raise ValueError("La distance ne peut pas être nulle si le potentiel ne l'est pas.")
    # Normalement, distance > 0, mais on gère le signe de V/d
    return différence_potentiel / distance

def calculer_xs(v0: float, theta: float, y0: float, q: float, m: float, E: float) -> float:
    """Calcule l'abscisse de contact xs."""
    # Cas spécial: Pas de champ électrique
    if abs(E) < 1e-15 or abs(q) < 1e-15 :
        # Si E ou q est nul, trajectoire rectiligne
        if abs(v0 * np.cos(theta)) < 1e-15: # Vitesse verticale nulle
            return np.inf # N'atteint jamais y=0 si part de y0 > 0
        temps_chute = y0 / (v0 * np.cos(theta)) # Attention: theta vs y, donc cos(theta) est comp. verticale
        return temps_chute * v0 * np.sin(theta) # sin(theta) est comp. horizontale

    # Calcul standard avec champ E non nul
    A = v0 * np.cos(theta)
    B = 2 * y0 * q * E / m
    C = q * E / m # accélération verticale / 2 (erreur formule originale, c'est ay = qE/m)
    # L'équation du mouvement est y(t) = y0 + v0*cos(theta)*t + 0.5*(q*E/m)*t^2
    # On cherche t quand y(t) = 0 => 0.5*ay*t^2 + vy0*t + y0 = 0
    ay = q * E / m
    vy0 = v0 * np.cos(theta)
    vx0 = v0 * np.sin(theta)

    # Résoudre 0.5*ay*t^2 + vy0*t + y0 = 0 pour t > 0
    discriminant_temps = vy0**2 - 4 * (0.5 * ay) * y0
    if discriminant_temps < 0:
        # Si ay > 0 (repulsion vers le haut) et vy0 > 0, ne redescend jamais
        # Si ay < 0 (attraction vers le bas), atteindra toujours y=0
        # Si ay=0, voir cas E=0 ci-dessus
        # Si ay > 0 et vy0 < 0, peut atteindre un minimum > 0
        # Le signe du discriminant dépend des signes relatifs de ay et y0
        # Si y0 > 0 et ay et vy0 sont tq la particule monte puis redescend mais sans atteindre 0
        if ay == 0: # Déjà traité mais pour être sûr
             return np.inf if vy0 >=0 else (y0 / (-vy0)) * vx0

        # Si ay != 0:
        # Minimum/maximum de y atteint quand vy(t) = vy0 + ay*t = 0 => t_extremum = -vy0/ay
        # y_extremum = y0 + vy0*t_extremum + 0.5*ay*t_extremum^2 = y0 - vy0^2/(2*ay)
        # Si y_extremum > 0 et ay > 0 (point bas > 0), ne touche jamais y=0
        y_extremum = y0 - (vy0**2) / (2 * ay) if ay != 0 else y0
        # Condition discriminant < 0 correspond à 2*ay*y0 > vy0^2
        # Si ay > 0 (repulsion vers le haut), et 2*ay*y0 > vy0^2, alors y reste > 0
        if ay > 0 and discriminant_temps < 0:
            # print(f"Debug: Disc < 0, ay={ay:.2e}, y0={y0:.2e}, vy0={vy0:.2e}, y_extremum={y_extremum:.2e}")
            raise ValueError("Discriminant négatif et ay > 0, la particule ne revient pas à y=0")
        # Si ay < 0 (attraction vers le bas), discriminant est toujours >= 0 car -2*ay*y0 > 0
        # Donc si discriminant < 0, c'est forcément que ay > 0

    # Si discriminant >= 0
    t1 = (-vy0 + np.sqrt(discriminant_temps)) / ay if ay != 0 else -y0/vy0 # Linéaire si ay=0
    t2 = (-vy0 - np.sqrt(discriminant_temps)) / ay if ay != 0 else -y0/vy0

    # On cherche la solution t > 0 physique
    temps_impact = np.nan
    if ay != 0:
        # Choisir la solution positive la plus pertinente
        valid_times = [t for t in [t1, t2] if t is not None and t > 1e-12] # Ignorer t quasi nul ou négatif
        if not valid_times:
             # Peut arriver si vy0=0, ay>0 (part de y0, repoussé vers le haut)
             if ay > 0 and abs(vy0) < 1e-12:
                  raise ValueError("Particule repoussée vers le haut depuis le repos, n'atteint pas y=0")
             else: # Autres cas ?
                  # print(f"Debug: Aucune solution t > 0 trouvée. t1={t1}, t2={t2}, ay={ay}, vy0={vy0}, y0={y0}")
                  raise ValueError("Pas de solution physique t > 0 pour l'impact")
        temps_impact = min(valid_times) # Prend le premier temps positif
    elif abs(vy0) > 1e-12 : # ay == 0
        temps_impact = -y0 / vy0
        if temps_impact <= 0:
             raise ValueError("Trajectoire rectiligne ne coupant pas y=0 pour t > 0")
    else: # ay=0 et vy0=0
         raise ValueError("Particule immobile (ay=0, vy0=0), n'atteint pas y=0")


    xs = vx0 * temps_impact
    return xs

def derivees_partielles(v0: float, theta: float, y0: float, q: float, m: float, E: float) -> tuple[float, float, float, float, float, float]:
    """Calcule les dérivées partielles de xs par rapport à chaque variable d'entrée."""
    # Utilise une différentiation numérique (plus simple et robuste aux formules complexes)
    h = 1e-8 # Petit pas

    # Calcul de référence
    try:
        xs_ref = calculer_xs(v0, theta, y0, q, m, E)
    except ValueError as e:
        raise ValueError(f"Impossible de calculer xs de référence: {e}")

    # dxs/dv0
    try:
        xs_v0_plus = calculer_xs(v0 + h, theta, y0, q, m, E)
        dxs_dv0 = (xs_v0_plus - xs_ref) / h
    except ValueError: dxs_dv0 = 0 # Ou np.nan, si la dérivée n'existe pas au point

    # dxs/dtheta
    try:
        xs_theta_plus = calculer_xs(v0, theta + h, y0, q, m, E)
        dxs_dtheta = (xs_theta_plus - xs_ref) / h
    except ValueError: dxs_dtheta = 0

    # dxs/dy0
    try:
        xs_y0_plus = calculer_xs(v0, theta, y0 + h, q, m, E)
        dxs_dy0 = (xs_y0_plus - xs_ref) / h
    except ValueError: dxs_dy0 = 0

    # dxs/dq
    try:
        # Utiliser un h relatif pour la charge qui peut être petite
        hq = h * abs(q) if abs(q) > 1e-15 else h
        xs_q_plus = calculer_xs(v0, theta, y0, q + hq, m, E)
        dxs_dq = (xs_q_plus - xs_ref) / hq
    except ValueError: dxs_dq = 0

    # dxs/dm
    try:
        hm = h * m # h relatif pour la masse
        xs_m_plus = calculer_xs(v0, theta, y0, q, m + hm, E)
        dxs_dm = (xs_m_plus - xs_ref) / hm
    except ValueError: dxs_dm = 0

    # dxs/dE
    try:
        # h relatif pour E
        hE = h * abs(E) if abs(E) > 1e-15 else h
        xs_E_plus = calculer_xs(v0, theta, y0, q, m, E + hE)
        dxs_dE = (xs_E_plus - xs_ref) / hE
    except ValueError: dxs_dE = 0

    return dxs_dv0, dxs_dtheta, dxs_dy0, dxs_dq, dxs_dm, dxs_dE

def calculer_incertitude(v0: float, theta: float, y0: float, q: float, m: float, E: float,
                           delta_v0: float, delta_theta: float, delta_y0: float,
                           delta_q: float, delta_m: float, delta_E: float) -> float:
    """Calcule l'incertitude totale sur xs (Δxs)."""
    try:
        dxs_dv0, dxs_dtheta, dxs_dy0, dxs_dq, dxs_dm, dxs_dE = derivees_partielles(v0, theta, y0, q, m, E)
    except ValueError as e:
         raise ValueError(f"Impossible de calculer les dérivées partielles: {e}")

    # Calcul de l'incertitude totale (somme quadratique)
    variance_xs = (
        (dxs_dv0 * delta_v0)**2 +
        (dxs_dtheta * delta_theta)**2 +
        (dxs_dy0 * delta_y0)**2 +
        (dxs_dq * delta_q)**2 +
        (dxs_dm * delta_m)**2 +
        (dxs_dE * delta_E)**2
    )
    # Gérer le cas où la variance est négative à cause d'erreurs numériques (rare)
    if variance_xs < 0: variance_xs = 0

    delta_xs = np.sqrt(variance_xs)
    return delta_xs

# --- Classe principale de l'application ---
class ParticleApp:
    def __init__(self, root):
        """
        Initialise l'application de simulation SIMS.
        """
        self.root = root
        self.root.title("Simulateur SIMS - Déviations")
        self.root.geometry("1600x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Style
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

        # Données
        self.particles_data = [] # Liste de tuples (masse_u: float, charge_e: float)

        # Structure Principale (PanedWindow)
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # Panneau de Contrôle Scrollable (Gauche)
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

        # Widgets dans le Panneau de Contrôle
        particle_frame = ttk.LabelFrame(control_panel, text="Gestion des Particules")
        particle_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_panel.columnconfigure(0, weight=1)
        self.create_particle_widgets(particle_frame)

        self.notebook = ttk.Notebook(control_panel)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        control_panel.rowconfigure(1, weight=1)

        self.mag_tab = ttk.Frame(self.notebook)
        self.elec_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')

        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)

        # Panneau Plot (Droite)
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

        # Barre de Statut
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Fonctions de Gestion Scrollbar ---
    def _bind_mousewheel(self, enter):
        if enter:
            self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.control_canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.control_canvas.bind_all("<Button-5>", self._on_mousewheel)
        else:
            self.control_canvas.unbind_all("<MouseWheel>")
            self.control_canvas.unbind_all("<Button-4>")
            self.control_canvas.unbind_all("<Button-5>")

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
            if content_height <= canvas_height:
                self.scrollbar.pack_forget()
            else:
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except tk.TclError: pass

    def _on_mousewheel(self, event):
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height: return
        except tk.TclError: return

        if event.num == 5 or event.delta < 0: self.control_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0: self.control_canvas.yview_scroll(-1, "units")
        return "break"

    # --- Fermeture Propre ---
    def _on_closing(self):
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le simulateur ?"):
            try: plt.close(self.fig)
            except Exception as e: print(f"Erreur fermeture Matplotlib: {e}")
            try: self.root.destroy()
            except Exception as e: print(f"Erreur destruction Tkinter: {e}")

    # --- Widgets Section Particules ---
    def create_particle_widgets(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill=tk.X)

        ttk.Label(input_frame, text="Masse (u):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mass_entry = ttk.Entry(input_frame, width=10)
        self.mass_entry.grid(row=0, column=1, padx=5, pady=2)
        self.mass_entry.insert(0, "1.0")

        ttk.Label(input_frame, text="Charge (e):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.charge_entry = ttk.Entry(input_frame, width=10)
        self.charge_entry.grid(row=0, column=3, padx=5, pady=2)
        self.charge_entry.insert(0, "1.0")

        add_btn = ttk.Button(input_frame, text="Ajouter", command=self.add_particle)
        add_btn.grid(row=0, column=4, padx=10, pady=2)

        ttk.Label(input_frame, text="Raccourcis :").grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))
        btns_frame = ttk.Frame(input_frame)
        btns_frame.grid(row=2, column=0, columnspan=5, pady=5, sticky="ew")
        num_btns = 3
        for i in range(num_btns): btns_frame.columnconfigure(i, weight=1)
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

        self.particle_tree = ttk.Treeview(tree_frame, columns=('Mass (u)', 'Charge (e)'), show='headings', height=6)
        self.particle_tree.heading('Mass (u)', text='Masse (u)')
        self.particle_tree.heading('Charge (e)', text='Charge (e)')
        self.particle_tree.column('Mass (u)', width=100, anchor=tk.CENTER)
        self.particle_tree.column('Charge (e)', width=100, anchor=tk.CENTER)
        scrollbar_tree = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar_tree.set)
        self.particle_tree.bind("<MouseWheel>", lambda e: self._on_mousewheel(e))
        self.particle_tree.bind("<Button-4>", lambda e: self._on_mousewheel(e))
        self.particle_tree.bind("<Button-5>", lambda e: self._on_mousewheel(e))
        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

        remove_btn = ttk.Button(parent, text="Supprimer Sélection", command=self.remove_particle)
        remove_btn.pack(pady=5, padx=10, fill=tk.X)

    # --- Fenêtre Tableau Périodique ---
    def ouvrir_fenetre_tp(self):
        if hasattr(self, 'molecule_fenetre') and self.molecule_fenetre.winfo_exists():
            self.molecule_fenetre.lift()
            return

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
            [], # Ligne vide
            [None, None, None, ('Ce', 140.12), ('Pr', 140.91), ('Nd', 144.24), ('Pm', 145.0), ('Sm', 150.36), ('Eu', 151.96), ('Gd', 157.25), ('Tb', 158.93), ('Dy', 162.50), ('Ho', 164.93), ('Er', 167.26), ('Tm', 168.93), ('Yb', 173.05), ('Lu', 174.97), None],
            [None, None, None, ('Th', 232.04), ('Pa', 231.04), ('U', 238.03), ('Np', 237.0), ('Pu', 244.0), ('Am', 243.0), ('Cm', 247.0), ('Bk', 247.0), ('Cf', 251.0), ('Es', 252.0), ('Fm', 257.0), ('Md', 258.0), ('No', 259.0), ('Lr', 262.0), None]
        ]

        table_frame = ttk.Frame(self.molecule_fenetre)
        table_frame.pack(pady=10, padx=10)

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

        ttk.Label(control_frame, text="Particule construite:", font=('Helvetica', 11, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0,5))
        display_reset_frame = ttk.Frame(control_frame)
        display_reset_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=5)
        display_reset_frame.columnconfigure(0, weight=1)
        self.molecule_display_var = tk.StringVar(value="(vide)")
        display_label = ttk.Label(display_reset_frame, textvariable=self.molecule_display_var, relief=tk.SUNKEN, padding=5, anchor=tk.W, font=('Consolas', 10))
        display_label.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))
        reset_btn = ttk.Button(display_reset_frame, text="Effacer", command=self.reset_molecule)
        reset_btn.grid(row=0, column=1, sticky=tk.E)

        charge_label = ttk.Label(control_frame, text="Charge (e):")
        charge_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.molecule_charge_var = tk.StringVar(value="1")
        charge_entry = ttk.Entry(control_frame, textvariable=self.molecule_charge_var, width=8)
        charge_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

        submit_btn = ttk.Button(control_frame, text="Ajouter cette Particule à la liste", command=self.submit_molecule)
        submit_btn.grid(row=3, column=0, columnspan=3, pady=10)

    def construction_de_molecule(self, symbol, mass):
        if symbol in self.selected_elts: self.selected_elts[symbol]['count'] += 1
        else: self.selected_elts[symbol] = {'mass': mass, 'count': 1}
        self._update_molecule_display()

    def reset_molecule(self):
        self.selected_elts = {}
        self._update_molecule_display()

    def _update_molecule_display(self):
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
        self._add_particle_to_list(mass_u, charge_e, f"Raccourci {mass_u:.3f} u")

    def submit_molecule(self):
        if not self.selected_elts:
            messagebox.showwarning("Aucun Élément", "Veuillez construire une particule.", parent=self.molecule_fenetre)
            return
        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())
            charge_str = self.molecule_charge_var.get().strip().replace(',', '.')
            if not charge_str: raise ValueError("Charge vide.")
            charge = float(charge_str)
            formula = self.molecule_display_var.get()
            added = self._add_particle_to_list(total_mass, charge, f"Particule {formula}")
            if added: self.molecule_fenetre.destroy()
        except ValueError as e:
            messagebox.showerror("Erreur de Saisie", f"Erreur : {e}", parent=self.molecule_fenetre)
            self.status_var.set("Erreur soumission molécule.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur inattendue: {e}", parent=self.molecule_fenetre)
            self.status_var.set("Erreur soumission molécule.")

    # --- Logique d'ajout et de suppression ---
    def add_particle(self):
        try:
            mass_u_str = self.mass_entry.get().strip().replace(',', '.')
            charge_e_str = self.charge_entry.get().strip().replace(',', '.')
            if not mass_u_str or not charge_e_str: raise ValueError("Masse et Charge vides.")
            mass_u = float(mass_u_str)
            charge_e = float(charge_e_str)
            self._add_particle_to_list(mass_u, charge_e, "Particule manuelle")
        except ValueError as e:
            messagebox.showerror("Erreur d'Entrée", f"Entrée invalide : {e}")
            self.status_var.set("Erreur ajout particule.")
        except Exception as e:
             messagebox.showerror("Erreur", f"Erreur inattendue: {e}")
             self.status_var.set("Erreur ajout particule.")

    def _add_particle_to_list(self, mass_u, charge_e, source_info=""):
        try:
            if mass_u <= 0: raise ValueError("Masse doit être > 0.")
            if charge_e == 0: raise ValueError("Charge ne peut pas être nulle.")

            particle_info = (round(mass_u, 5), round(charge_e, 5))
            existing_particles = [(round(p[0], 5), round(p[1], 5)) for p in self.particles_data]

            if particle_info not in existing_particles:
                self.particles_data.append((mass_u, charge_e))
                self.particle_tree.insert('', tk.END, values=(f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                self.status_var.set(f"{source_info} ajoutée: {mass_u:.3f} u, {charge_e:+.2f} e")
                # Mettre à jour le défilement si nécessaire après ajout
                self.root.after(50, self._update_scroll_region_and_bar)
                return True
            else:
                parent_win = getattr(self, 'molecule_fenetre', self.root)
                if parent_win and parent_win.winfo_exists():
                    messagebox.showwarning("Doublon", f"Particule {mass_u:.3f} u / {charge_e:+.2f} e déjà listée.", parent=parent_win)
                else:
                     messagebox.showwarning("Doublon", f"Particule {mass_u:.3f} u / {charge_e:+.2f} e déjà listée.")
                self.status_var.set("Ajout annulé (doublon).")
                return False
        except ValueError as e:
            parent_win = getattr(self, 'molecule_fenetre', self.root)
            if parent_win and parent_win.winfo_exists(): messagebox.showerror("Erreur Validation", f"{e}", parent=parent_win)
            else: messagebox.showerror("Erreur Validation", f"{e}")
            self.status_var.set(f"Erreur validation: {e}")
            return False
        except Exception as e:
             parent_win = getattr(self, 'molecule_fenetre', self.root)
             if parent_win and parent_win.winfo_exists(): messagebox.showerror("Erreur Inattendue", f"{e}", parent=parent_win)
             else: messagebox.showerror("Erreur Inattendue", f"{e}")
             self.status_var.set("Erreur interne lors de l'ajout.")
             return False

    def remove_particle(self):
        selected_items = self.particle_tree.selection()
        if not selected_items:
            messagebox.showwarning("Aucune Sélection", "Sélectionnez une ou plusieurs particules à supprimer.")
            return
        if len(selected_items) > 1:
            if not messagebox.askyesno("Confirmation", f"Supprimer les {len(selected_items)} particules sélectionnées ?"):
                return

        indices_to_remove = []
        items_to_remove_tree = []
        for item_id in selected_items:
            try:
                index = self.particle_tree.index(item_id)
                indices_to_remove.append(index)
                items_to_remove_tree.append(item_id)
            except tk.TclError: print(f"Avertissement: Item {item_id} non trouvé.")

        indices_to_remove.sort(reverse=True)
        deleted_count = 0
        for index in indices_to_remove:
            try:
                del self.particles_data[index]
                deleted_count += 1
            except IndexError: print(f"Avertissement: Index {index} hors limites.")

        for item_id in items_to_remove_tree:
            if self.particle_tree.exists(item_id): self.particle_tree.delete(item_id)

        self.status_var.set(f"{deleted_count} particule(s) supprimée(s).")
        # Mettre à jour le défilement après suppression
        self.root.after(50, self._update_scroll_region_and_bar)

    # --- Widgets Magnétiques ---
    def create_magnetic_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.dynamic_inputs_frame = ttk.Frame(frame)
        self.base_inputs_frame = ttk.Frame(frame)
        self.x_detecteur_var = tk.StringVar(value="0.1")
        self.add_labeled_entry(frame, "X détecteur (m):", self.x_detecteur_var).pack(fill=tk.X, pady=3)
        self.dynamic_trace_var = tk.BooleanVar(value=False)
        dynamic_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_trace_var, command=self.toggle_dynamic_inputs)
        dynamic_check.pack(anchor=tk.W, pady=5)
        # Mode Statique
        parent_base = self.base_inputs_frame
        self.v0_mag_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_mag_var).pack(fill=tk.X, pady=3)
        self.bz_mag_var = tk.StringVar(value="0.2")
        self.add_labeled_entry(parent_base, "Champ Magnétique (T):", self.bz_mag_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_magnetic_simulation)
        trace_btn_base.pack(pady=15)
        # Mode Dynamique
        parent_dyn = self.dynamic_inputs_frame
        self.bz_min_var = tk.StringVar(value="0.01")
        self.add_labeled_entry(parent_dyn, "Bz min (T):", self.bz_min_var).pack(fill=tk.X, pady=3)
        self.bz_max_var = tk.StringVar(value="0.5")
        self.add_labeled_entry(parent_dyn, "Bz max (T):", self.bz_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Champ Magnétique Bz (T):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_bz = ttk.Frame(parent_dyn)
        self.slider_frame_bz.pack(fill=tk.X, pady=(0,5))
        self.bz_var = tk.DoubleVar(value=0.2)
        self.bz_slider = ttk.Scale(self.slider_frame_bz, from_=0.01, to=0.5, orient=tk.HORIZONTAL, variable=self.bz_var, command=self._on_bz_slider_change)
        self.bz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.bz_label_var = tk.StringVar(value=f"{self.bz_var.get():.3f} T")
        ttk.Label(self.slider_frame_bz, textvariable=self.bz_label_var, width=10).pack(side=tk.LEFT)
        self.v0_min_var = tk.StringVar(value="1e5")
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.v0_min_var).pack(fill=tk.X, pady=3)
        self.v0_max_var = tk.StringVar(value="1e6")
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.v0_max_var).pack(fill=tk.X, pady=3)
        ttk.Label(parent_dyn, text="Vitesse Initiale (m/s):").pack(anchor=tk.W, pady=(5,0))
        self.slider_frame_v0 = ttk.Frame(parent_dyn)
        self.slider_frame_v0.pack(fill=tk.X, pady=(0,5))
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
            self.base_inputs_frame.pack_forget()
            self.dynamic_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else:
            self.dynamic_inputs_frame.pack_forget()
            self.base_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        # Mettre à jour le défilement lors du changement d'onglet/mode
        self.root.after(50, self._update_scroll_region_and_bar)


    # --- Callbacks Sliders Magnétiques ---
    def _on_bz_slider_change(self, event=None):
        self._update_bz_label()
        if self.particles_data: self.run_magnetic_simulation(called_by_slider=True)
    def _update_bz_label(self, event=None): self.bz_label_var.set(f"{self.bz_var.get():.3f} T")
    def _on_v0_slider_change(self, event=None):
        self._update_v0_label()
        if self.particles_data: self.run_magnetic_simulation(called_by_slider=True)
    def _update_v0_label(self, event=None): self.v0_label_var.set(f"{self.v0_var.get():.2e} m/s")

    # --- Widgets Électriques ---
    def create_electric_widgets(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        self.dynamic_electric_inputs_frame = ttk.Frame(frame)
        self.base_electric_inputs_frame = ttk.Frame(frame)
        # Widgets Communs
        self.angle_var = tk.StringVar(value="30") # Degrés
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3)
        self.dist_var = tk.StringVar(value="0.05") # m
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3)

        # Checkbox Mode Dynamique
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        dynamic_elec_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)", variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric)
        dynamic_elec_check.pack(anchor=tk.W, pady=5)

        # Mode Statique
        parent_base = self.base_electric_inputs_frame
        self.v0_elec_var = tk.StringVar(value="1e5") # m/s
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)
        self.diff_pot_var = tk.StringVar(value="-5000") # Volts
        self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_electric_simulation)
        trace_btn_base.pack(pady=15)

        # Mode Dynamique
        parent_dyn = self.dynamic_electric_inputs_frame
        # Limites V0
        self.elec_v0_min_var = tk.StringVar(value="1e4")
        self.add_labeled_entry(parent_dyn, "V0 min (m/s):", self.elec_v0_min_var).pack(fill=tk.X, pady=3)
        self.elec_v0_max_var = tk.StringVar(value="2e5")
        self.add_labeled_entry(parent_dyn, "V0 max (m/s):", self.elec_v0_max_var).pack(fill=tk.X, pady=3)
        # Slider V0
        ttk.Label(parent_dyn, text="Vitesse Initiale V0 (m/s):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v0_elec = ttk.Frame(parent_dyn)
        self.slider_frame_v0_elec.pack(fill=tk.X, pady=(0, 5))
        self.v0_var_elec = tk.DoubleVar(value=1.05e5)
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=1e4, to=2e5, orient=tk.HORIZONTAL, variable=self.v0_var_elec, command=self._on_v0_slider_change_elec)
        self.v0_slider_elec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.v0_label_var_elec = tk.StringVar(value=f"{self.v0_var_elec.get():.2e} m/s")
        ttk.Label(self.slider_frame_v0_elec, textvariable=self.v0_label_var_elec, width=12).pack(side=tk.LEFT)
        # Limites Potentiel
        self.diff_pot_min_var = tk.StringVar(value="-10000")
        self.add_labeled_entry(parent_dyn, "Potentiel min (V):", self.diff_pot_min_var).pack(fill=tk.X, pady=3)
        self.diff_pot_max_var = tk.StringVar(value="10000")
        self.add_labeled_entry(parent_dyn, "Potentiel max (V):", self.diff_pot_max_var).pack(fill=tk.X, pady=3)
        # Slider Potentiel
        ttk.Label(parent_dyn, text="Diff. Potentiel (V):").pack(anchor=tk.W, pady=(5, 0))
        self.slider_frame_v = ttk.Frame(parent_dyn)
        self.slider_frame_v.pack(fill=tk.X, pady=(0, 5))
        self.pot_var = tk.DoubleVar(value=0)
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL, variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V")
        ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=12).pack(side=tk.LEFT)
        # Bouton Appliquer Limites
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_electric_simulation)
        apply_limits_btn_dyn.pack(pady=15)

        # --- Section Incertitude ---
        uncertainty_separator = ttk.Separator(frame, orient=tk.HORIZONTAL)
        uncertainty_separator.pack(fill=tk.X, pady=10, padx=5)

        self.show_uncertainty_var = tk.BooleanVar(value=False)
        uncertainty_check = ttk.Checkbutton(frame, text="Afficher Incertitude sur xs",
                                            variable=self.show_uncertainty_var,
                                            command=self.toggle_uncertainty_inputs)
        uncertainty_check.pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Frame pour les entrées d'incertitude
        self.uncertainty_inputs_frame = ttk.LabelFrame(frame, text="Paramètres d'incertitude")
        # Ne pas l'afficher initialement, sera géré par toggle_uncertainty_inputs

        # Entrées pour les deltas (valeurs par défaut raisonnables)
        self.delta_v0_percent_var = tk.StringVar(value="1.0") # %
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔV0 (%):", self.delta_v0_percent_var).pack(fill=tk.X, pady=2, padx=5)

        self.delta_theta_deg_var = tk.StringVar(value="0.5") # degrés absolus
        self.add_labeled_entry(self.uncertainty_inputs_frame, "Δθ (degrés):", self.delta_theta_deg_var).pack(fill=tk.X, pady=2, padx=5)

        self.delta_y0_percent_var = tk.StringVar(value="2.0") # % de la hauteur initiale
        self.add_labeled_entry(self.uncertainty_inputs_frame, "Δy0 (%):", self.delta_y0_percent_var).pack(fill=tk.X, pady=2, padx=5)

        self.delta_potentiel_percent_var = tk.StringVar(value="1.0") # % du potentiel
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔPotentiel (%):", self.delta_potentiel_percent_var).pack(fill=tk.X, pady=2, padx=5)

        self.delta_hauteur_percent_var = tk.StringVar(value="1.0") # % de la distance/hauteur pour E
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔHauteur (pour E) (%):", self.delta_hauteur_percent_var).pack(fill=tk.X, pady=2, padx=5)

        ttk.Label(self.uncertainty_inputs_frame, text="Note: Δm/m = 0.1%, Δq/q = 0.01% (fixes)", font=('Segoe UI', 8)).pack(pady=(5,0))
        # --- Fin Section Incertitude ---


        # Afficher/Cacher initialement les widgets
        self.toggle_dynamic_electric() # Gère les widgets dynamique/statique
        self.toggle_uncertainty_inputs() # Gère les widgets d'incertitude

    def toggle_dynamic_electric(self):
        if self.dynamic_elec_var.get():
            self.base_electric_inputs_frame.pack_forget()
            self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else:
            self.dynamic_electric_inputs_frame.pack_forget()
            self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        # Mettre à jour le défilement
        self.root.after(50, self._update_scroll_region_and_bar)

    def toggle_uncertainty_inputs(self):
        """Affiche ou cache le frame des entrées d'incertitude."""
        if self.show_uncertainty_var.get():
            self.uncertainty_inputs_frame.pack(fill=tk.X, pady=5, padx=5, before=self.base_electric_inputs_frame if not self.dynamic_elec_var.get() else self.dynamic_electric_inputs_frame)
        else:
            self.uncertainty_inputs_frame.pack_forget()
        # Mettre à jour le défilement
        self.root.after(50, self._update_scroll_region_and_bar)


    # --- Callbacks Sliders Électriques ---
    def _on_pot_slider_change(self, event=None):
        self._update_pot_label()
        if self.particles_data: self.run_electric_simulation(called_by_slider=True)
    def _on_v0_slider_change_elec(self, event=None):
        self._update_v0_label_elec()
        if self.particles_data: self.run_electric_simulation(called_by_slider=True)
    def _update_pot_label(self, event=None): self.pot_label_var.set(f"{self.pot_var.get():.1f} V")
    def _update_v0_label_elec(self, event=None): self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e} m/s")

    # --- Helper ---
    def add_labeled_entry(self, parent, label_text, string_var):
        entry_frame = ttk.Frame(parent)
        ttk.Label(entry_frame, text=label_text, width=22, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return entry_frame

    # --- Exécution des Simulations ---
    def run_magnetic_simulation(self, called_by_slider=False):
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla()
            self.ax.set_title("Déviation Magnétique")
            self.ax.set_xlabel("Position x (m)")
            self.ax.set_ylabel("Position y (m)")
            self.canvas.draw()
            return
        try:
            x_detecteur_str = self.x_detecteur_var.get().strip().replace(',', '.')
            if not x_detecteur_str: raise ValueError("X détecteur vide.")
            x_detecteur = float(x_detecteur_str)
            if x_detecteur <= 0: raise ValueError("X détecteur doit être > 0.")

            if not self.dynamic_trace_var.get(): # Mode Statique
                v0_str = self.v0_mag_var.get().strip().replace(',', '.')
                bz_str = self.bz_mag_var.get().strip().replace(',', '.')
                if not v0_str or not bz_str: raise ValueError("V0 et Bz vides.")
                v0 = float(v0_str); bz = float(bz_str)
                if v0 <= 0: raise ValueError("V0 doit être > 0.")
                if abs(bz) < 1e-15: raise ValueError("Bz ne peut être nul.")
            else: # Mode Dynamique
                if not called_by_slider:
                    bz_min = float(self.bz_min_var.get().strip().replace(',', '.'))
                    bz_max = float(self.bz_max_var.get().strip().replace(',', '.'))
                    v0_min = float(self.v0_min_var.get().strip().replace(',', '.'))
                    v0_max = float(self.v0_max_var.get().strip().replace(',', '.'))
                    if v0_min <= 0 : raise ValueError("V0 min doit être > 0.")
                    if v0_min >= v0_max : raise ValueError("V0 max > V0 min requis.")
                    if bz_min >= bz_max : raise ValueError("Bz max > Bz min requis.")

                    current_bz = self.bz_var.get(); current_v0 = self.v0_var.get()
                    self.bz_slider.config(from_=bz_min, to=bz_max)
                    if not (bz_min <= current_bz <= bz_max): self.bz_var.set(max(min(current_bz, bz_max), bz_min)) # Clamp
                    self._update_bz_label()
                    self.v0_slider.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max): self.v0_var.set(max(min(current_v0, v0_max), v0_min)) # Clamp
                    self._update_v0_label()

                v0 = self.v0_var.get(); bz = self.bz_var.get()
                if abs(bz) < 1e-15: raise ValueError("Bz est trop proche de zéro.")

            self.ax.cla()
            self.status_var.set("Calcul déviation magnétique...")
            self.root.update_idletasks()
            partie_electroaimant.tracer_ensemble_trajectoires(
                self.particles_data, v0, bz, x_detecteur, create_plot=False, ax=self.ax)
            self.ax.relim(); self.ax.autoscale_view(True, True, True)
            self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")
        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Magnétique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Mag): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur (Magnétique):\n{e}", parent=self.root)
            print(f"Erreur Simulation Magnétique: {type(e).__name__}: {e}")
            self.status_var.set("Erreur simulation magnétique.")

# (Le début de la classe ParticleApp et les autres méthodes restent inchangés)
# ...

    def run_electric_simulation(self, called_by_slider=False):
        if not self.particles_data:
            # ... (gestion si pas de particules - inchangé)
            return

        try:
            # --- Paramètres principaux ---
            # ... (lecture angle, distance, v0, potentiel - inchangé)
            angle_deg = float(self.angle_var.get().strip().replace(',', '.'))
            hauteur_distance = float(self.dist_var.get().strip().replace(',', '.'))
            if hauteur_distance <= 1e-9: raise ValueError("Hauteur/Distance doit être > 0.")
            if not (0 < angle_deg < 90): raise ValueError("Angle doit être > 0° et < 90°.")
            angle_rad = np.radians(angle_deg)
            hauteur_initiale = hauteur_distance # y0 = hauteur

            if not self.dynamic_elec_var.get(): # Mode Statique
                v0 = float(self.v0_elec_var.get().strip().replace(',', '.'))
                potentiel = float(self.diff_pot_var.get().strip().replace(',', '.'))
                
                if v0 <= 0 : raise ValueError("V0 doit être > 0.")
            else: # Mode Dynamique
                # ... (mise à jour et lecture des sliders - inchangé)
                v0 = self.v0_var_elec.get()
                potentiel = self.pot_var.get()

            # --- Calcul Incertitude (si demandé) ---
            show_uncertainty = self.show_uncertainty_var.get()
            # uncertainty_results stocke le résultat du *calcul* (pas encore la décision de tracer)
            uncertainty_results = {} # Dict: index -> {'xs': float, 'delta_xs': float} | None
            if show_uncertainty:
                try:
                    # ... (lecture des deltas depuis GUI - inchangé)
                    delta_v0_percent = float(self.delta_v0_percent_var.get().strip().replace(',', '.'))
                    # ... (autres deltas)
                    delta_hauteur_percent = float(self.delta_hauteur_percent_var.get().strip().replace(',', '.'))

                    # ... (Calcul E et delta_E - inchangé)
                    E = champ_electrique_v2(hauteur_distance, potentiel)
                    delta_d_for_E = hauteur_distance * (delta_hauteur_percent / 100.0)
                    delta_V = abs(potentiel) * (float(self.delta_potentiel_percent_var.get().strip().replace(',', '.')) / 100.0) # Lire delta_potentiel ici
                    delta_E = 0.0
                    term_V_sq = (delta_V / potentiel)**2 if abs(potentiel) > 1e-15 else 0
                    term_d_sq = (delta_d_for_E / hauteur_distance)**2
                    if abs(E) > 1e-15:
                       delta_E = abs(E) * np.sqrt(term_V_sq + term_d_sq)


                    # Boucle sur les particules pour *calculer* xs et delta_xs
                    for i, (mass_u, charge_e) in enumerate(self.particles_data):
                        try:
                            m_si = mass_u * constants.u
                            q_si = charge_e * constants.e

                            # ... (Calcul deltas absolus: delta_v0_abs, delta_theta_rad, etc. - inchangé)
                            delta_v0_abs = v0 * (delta_v0_percent / 100.0)
                            delta_theta_rad = np.radians(float(self.delta_theta_deg_var.get().strip().replace(',', '.')))
                            delta_y0_abs = hauteur_initiale * (float(self.delta_y0_percent_var.get().strip().replace(',', '.')) / 100.0)
                            delta_m_abs = m_si * 0.001
                            delta_q_abs = abs(q_si) * 0.0001

                            # Calculer xs et delta_xs
                            current_xs = calculer_xs(v0, angle_rad, hauteur_initiale, q_si, m_si, E)
                            current_delta_xs = calculer_incertitude(v0, angle_rad, hauteur_initiale, q_si, m_si, E,
                                                                    delta_v0_abs, delta_theta_rad, delta_y0_abs,
                                                                    delta_q_abs, delta_m_abs, delta_E)
                            uncertainty_results[i] = {'xs': current_xs, 'delta_xs': current_delta_xs}

                        except ValueError as e_particle:
                            # Note: On calcule quand même, même si la formule analytique dit non-impact.
                            # La décision de tracer sera basée sur la simulation.
                            # Mais si le *calcul d'incertitude* échoue, on ne pourra pas tracer.
                            print(f"Warning: Incertitude non calculable pour particule {i} ({mass_u:.2f}u): {e_particle}")
                            uncertainty_results[i] = None # Marquer comme échec du calcul
                        except Exception as e_gen:
                             print(f"Erreur calcul incertitude particule {i}: {type(e_gen).__name__} {e_gen}")
                             uncertainty_results[i] = None

                except ValueError as e_delta:
                     messagebox.showerror("Erreur Incertitude", f"Valeur d'incertitude invalide : {e_delta}", parent=self.root)
                     show_uncertainty = False
                     self.status_var.set("Erreur paramètre incertitude.")
                except Exception as e_unc_glob:
                     messagebox.showerror("Erreur Incertitude", f"Erreur calcul incertitude : {e_unc_glob}", parent=self.root)
                     show_uncertainty = False
                     self.status_var.set("Erreur calcul incertitude.")

            # --- Tracé ---
            self.ax.cla()
            self.status_var.set("Calcul déviation électrique...")
            self.root.update_idletasks()

            # Appel backend pour tracer les trajectoires principales
            # !! Idéalement, cette fonction retournerait des infos sur l'impact !!
            # Ex: simulation_results = deviation.tracer_ensemble_trajectoires(...)
            deviation.tracer_ensemble_trajectoires(
                self.particles_data, v0, potentiel, angle_rad, hauteur_initiale,
                create_plot=False, ax=self.ax
            )

            # Récupérer les lignes tracées et leurs couleurs
            # Attention: S'assurer que l'ordre correspond aux particules_data
            lines = self.ax.get_lines()
            colors = [line.get_color() for line in lines]
            num_particles = len(self.particles_data)
            num_lines = len(lines)

            # Vérifier si on peut faire correspondre lignes et particules
            can_check_impact = (num_lines >= num_particles)
            if not can_check_impact and show_uncertainty:
                print("Warning: Moins de lignes tracées que de particules. Impossible de vérifier l'impact pour l'incertitude.")

            # Ajouter les lignes d'incertitude UNIQUEMENT si la simulation a atteint y=0
            if show_uncertainty:
                plotted_uncertainty_legend = False
                impact_threshold = 1e-6 # Seuil pour considérer y=0 atteint

                for i in range(num_particles):
                    # 1. Vérifier si le calcul d'incertitude a réussi pour cette particule
                    unc_calc_result = uncertainty_results.get(i)
                    if unc_calc_result and np.isfinite(unc_calc_result['xs']) and np.isfinite(unc_calc_result['delta_xs']):
                        # 2. Vérifier si la *trajectoire simulée* a atteint y=0
                        sim_trajectory_hit = False
                        if can_check_impact:
                             # Prendre la ligne correspondante (en espérant que l'ordre est bon)
                             line = lines[i]
                             y_data = line.get_ydata()
                             if len(y_data) > 0 and np.min(y_data) < impact_threshold:
                                 sim_trajectory_hit = True

                        # 3. Si les deux conditions sont vraies, tracer l'incertitude
                        if sim_trajectory_hit:
                            xs = unc_calc_result['xs']
                            delta_xs = unc_calc_result['delta_xs']
                            # Utiliser la couleur de la ligne correspondante
                            color = colors[i % len(colors)] # Utiliser modulo au cas où len(colors) < num_particles
                            line_style = '--'
                            line_alpha = 0.7
                            label = "Plage d'incertitude (xs ± Δxs)" if not plotted_uncertainty_legend else ""

                            self.ax.axvline(xs - delta_xs, color=color, linestyle=line_style, alpha=line_alpha, label=label)
                            self.ax.axvline(xs + delta_xs, color=color, linestyle=line_style, alpha=line_alpha)
                            plotted_uncertainty_legend = True

                # Mettre à jour la légende pour inclure l'entrée d'incertitude (si ajoutée)
                if plotted_uncertainty_legend:
                    handles, labels = self.ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    self.ax.legend(by_label.values(), by_label.keys())
                else:
                     # Si une légende existait déjà (créée par deviation.py?), la reafficher telle quelle
                     current_legend = self.ax.get_legend()
                     if current_legend:
                         self.ax.legend()


            # Finaliser le plot
            self.ax.autoscale_view(scalex=True, scaley=False) # Réajuster X si axvline a élargi
            self.canvas.draw()

            # Mise à jour barre de statut
            # ... (logique de message de statut - inchangé, mais reflète maintenant la condition de tracé)
            status_msg = "Tracé déviation électrique terminé."
            if show_uncertainty:
                status_msg += " (Avec incertitude si impact)." # Message simplifié
            self.status_var.set(status_msg)


        except ValueError as e:
            # ... (gestion des erreurs de paramètres - inchangé)
             if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Électrique):\n{e}", parent=self.root)
             self.status_var.set(f"Erreur paramètre (Elec): {e}")
        except Exception as e:
            # ... (gestion des erreurs générales - inchangé)
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur (Électrique):\n{e}", parent=self.root)
            import traceback
            print(f"Erreur Simulation Électrique: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.status_var.set("Erreur simulation électrique.")


# --- Point d'entrée ---
# ... (inchangé)
if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()