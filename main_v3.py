# -*- coding: utf-8 -*- # Pour assurer la compatibilité des caractères spéciaux (ex: accents)
import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration des chemins ---
folder = os.path.dirname(os.path.abspath(__file__))
path_partie_bleue = os.path.join(folder, "SIMS","Partie Bleue (accélération)", "Code")
path_partie_verte = os.path.join(folder, "SIMS","Partie Verte (déviation magnétique)", "Code")
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
        self.root.title("Simulateur SIMS - Déviations (v2)") # Titre mis à jour
        self.root.geometry("1600x800")
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
        self.particles_data = [] # Liste de tuples (masse_u: float, charge_e: float)

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
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self._update_scroll_region_and_bar(e)
        )

        # Placer le frame interne dans le Canvas
        self.window_id = self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Lier la scrollbar au Canvas
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Placement du Canvas et de la Scrollbar dans leur conteneur
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

        self.notebook.add(self.mag_tab, text='Déviation Magnétique')
        self.notebook.add(self.elec_tab, text='Déviation Électrique')

        self.create_magnetic_widgets(self.mag_tab)
        self.create_electric_widgets(self.elec_tab)

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
        self.root.after(10, self._check_and_set_scrollbar_state)

    def _check_and_set_scrollbar_state(self):
        """Vérifie et active/désactive la scrollbar."""
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height:
                # Cacher la scrollbar si pas nécessaire
                self.scrollbar.pack_forget()
            else:
                # Afficher la scrollbar si nécessaire
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except tk.TclError:
            pass # Fenêtre en cours de fermeture

    def _on_mousewheel(self, event):
        """Gère le défilement par molette si scroll nécessaire."""
        # Déterminer si on doit scroller
        try:
            canvas_height = self.control_canvas.winfo_height()
            content_height = self.scrollable_frame.winfo_reqheight()
            if content_height <= canvas_height:
                return # Pas besoin de scroller
        except tk.TclError:
             return # Erreur si fenêtre fermée

        # Défiler
        if event.num == 5 or event.delta < 0:
            self.control_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.control_canvas.yview_scroll(-1, "units")
        return "break" # Empêcher la propagation si on a scrollé ici


    # --- Fermeture Propre ---
    def _on_closing(self):
        """Gère la fermeture de la fenêtre."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le simulateur ?"):
            try: plt.close(self.fig)
            except Exception as e: 
                print(f"Erreur fermeture Matplotlib: {e}")
            try:
                self.root.destroy()
            except Exception as e: print(f"Erreur destruction Tkinter: {e}")

    # --- Widgets Section Particules ---
    def create_particle_widgets(self, parent):
        """
        Crée les widgets pour ajouter, lister et supprimer des particules.

        Parameters
        ----------
        parent : ttk.LabelFrame
            Le conteneur (Frame) dans lequel placer les widgets.
        """
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

        # --- Section Raccourcis ---
        ttk.Label(input_frame, text="Raccourcis :").grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))

        btns_frame = ttk.Frame(input_frame)
        # Utiliser grid pour que ça prenne la largeur
        btns_frame.grid(row=2, column=0, columnspan=5, pady=5, sticky="ew")

        # Configurer les colonnes pour un espacement égal
        num_btns = 3
        for i in range(num_btns):
            btns_frame.columnconfigure(i, weight=1)

        # Exemples de particules communes en SIMS
        btn_o2 = ttk.Button(btns_frame, text="O₂⁻", command=lambda: self.ajt_particle_connue(31.998, -1.0))
        btn_o2.grid(row=0, column=0, padx=2, sticky="ew")

        btn_si = ttk.Button(btns_frame, text="Si⁺", command=lambda: self.ajt_particle_connue(28.085, +1.0))
        btn_si.grid(row=0, column=1, padx=2, sticky="ew")

        btn_h = ttk.Button(btns_frame, text="H⁺", command=lambda: self.ajt_particle_connue(1.008, +1.0))
        btn_h.grid(row=0, column=2, padx=2, sticky="ew")

        # Bouton pour ouvrir le constructeur de molécules
        create_molecule_btn = ttk.Button(parent, text="Construire une Particule...", command=self.ouvrir_fenetre_tp)
        create_molecule_btn.pack(pady=(5, 10), padx=10, fill=tk.X)

        # --- Liste des Particules (Treeview) ---
        tree_frame = ttk.Frame(parent)
        # Augmenter la hauteur visible par défaut
        tree_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True, ipady=10)

        self.particle_tree = ttk.Treeview(tree_frame, columns=('Mass (u)', 'Charge (e)'), show='headings', height=6) # Hauteur augmentée
        self.particle_tree.heading('Mass (u)', text='Masse (u)')
        self.particle_tree.heading('Charge (e)', text='Charge (e)')
        self.particle_tree.column('Mass (u)', width=100, anchor=tk.CENTER) # Plus large
        self.particle_tree.column('Charge (e)', width=100, anchor=tk.CENTER) # Plus large

        # Scrollbar pour le Treeview
        scrollbar_tree = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.particle_tree.yview)
        self.particle_tree.configure(yscrollcommand=scrollbar_tree.set)

        # Important: Lier la molette aussi au Treeview s'il a le focus
        self.particle_tree.bind("<MouseWheel>", lambda e: self._on_mousewheel(e))
        self.particle_tree.bind("<Button-4>", lambda e: self._on_mousewheel(e))
        self.particle_tree.bind("<Button-5>", lambda e: self._on_mousewheel(e))

        self.particle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)
        # --- Fin Liste Particules ---

        # Bouton Supprimer
        remove_btn = ttk.Button(parent, text="Supprimer Sélection", command=self.remove_particle)
        remove_btn.pack(pady=5, padx=10, fill=tk.X)

    # --- Fenêtre Tableau Périodique ---
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

        # Configuration pour centrer les éléments sur 3 colonnes
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=0)
        control_frame.columnconfigure(2, weight=1)

        # Label "Particule construite"
        ttk.Label(control_frame, text="Particule construite:", font=('Helvetica', 14, 'bold')).grid(row=0, column=1, sticky="", pady=(0,5))

        # Frame pour affichage et reset
        display_reset_frame = ttk.Frame(control_frame)
        display_reset_frame.grid(row=1, column=1, sticky="ew", pady=5)
        display_reset_frame.columnconfigure(0, weight=1)

        self.molecule_display_var = tk.StringVar(value="(vide)")
        # Label avec bordure pour mieux le voir
        display_label = ttk.Label(display_reset_frame, textvariable=self.molecule_display_var,
                                   relief=tk.SUNKEN, padding=5, anchor=tk.W, font=('Consolas', 10))
        display_label.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))

        reset_btn = ttk.Button(display_reset_frame, text="Effacer", command=self.reset_molecule)
        reset_btn.grid(row=0, column=1, sticky=tk.E)

        # Charge
        charge_frame = ttk.Frame(control_frame)
        charge_frame.grid(row=2, column=1, sticky="", pady=5)

        charge_label = ttk.Label(charge_frame, text="Charge (e):")
        charge_label.pack(side=tk.LEFT)
        self.molecule_charge_var = tk.StringVar(value="1")
        charge_entry = ttk.Entry(charge_frame, textvariable=self.molecule_charge_var, width=8)
        charge_entry.pack(side=tk.LEFT, padx=5)

        # Bouton Soumettre
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

        # Trier par symbole pour une formule conventionnelle 
        sorted_symbols = sorted(self.selected_elts.keys())

        molecule_parts = []
        for symbol in sorted_symbols:
            data = self.selected_elts[symbol]
            count = data['count']
            part = symbol
            if count > 1:
                # Utilisation d'indices Unicode pour une jolie formule
                subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                part += str(count).translate(subscript_map)
            molecule_parts.append(part)

        self.molecule_display_var.set("".join(molecule_parts)) # Joindre sans espace

    def ajt_particle_connue(self, mass_u, charge_e):
        """Ajoute une particule prédéfinie (O₂, Si, H...) directement."""
        self._add_particle_to_list(mass_u, charge_e, f"Raccourci {mass_u:.3f} u")

    def submit_molecule(self):
        """Calcule la masse, lit la charge, et ajoute la particule construite."""
        if not self.selected_elts:
            messagebox.showwarning("Aucun Élément", "Veuillez cliquer sur les éléments du tableau pour construire votre particule.", parent=self.molecule_fenetre)
            return

        try:
            total_mass = sum(v['mass'] * v['count'] for v in self.selected_elts.values())
            charge_str = self.molecule_charge_var.get().strip().replace(',', '.') # Nettoyer entrée
            if not charge_str: raise ValueError("La charge ne peut pas être vide.")
            charge = float(charge_str) # Peut lever ValueError

            formula = self.molecule_display_var.get() # Récupérer la formule affichée

            # Appeler la fonction interne d'ajout
            added = self._add_particle_to_list(total_mass, charge, f"Particule {formula}")

            # Fermer la fenêtre seulement si l'ajout a réussi
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
            if not mass_u_str or not charge_e_str:
                 raise ValueError("Masse et Charge ne peuvent être vides.")
            mass_u = float(mass_u_str)
            charge_e = float(charge_e_str)

            # Appeler la fonction interne d'ajout
            self._add_particle_to_list(mass_u, charge_e, "Particule manuelle")

            # Vider les champs après ajout réussi
            self.mass_entry.delete(0, tk.END)
            self.charge_entry.delete(0, tk.END)

        except ValueError as e:
            messagebox.showerror("Erreur d'Entrée", f"Entrée invalide : {e}")
            self.status_var.set("Erreur d'ajout de particule.")
        except Exception as e:
             messagebox.showerror("Erreur", f"Une erreur inattendue est survenue: {e}")
             self.status_var.set("Erreur d'ajout de particule.")

    def _add_particle_to_list(self, mass_u, charge_e, source_info=""):
        """
        Fonction interne pour valider et ajouter une particule à la liste et au Treeview.
        Retourne True si ajoutée, False sinon.
        """
        try:
            if mass_u <= 0:
                raise ValueError("Masse doit être > 0.")
            if charge_e == 0:
                raise ValueError("Charge ne peut pas être nulle pour la déviation.")

            # --- Vérification de signe (particules doivent avoir une charge de même signe) ---
            current_sign = np.sign(charge_e)
            if len(self.particles_data) > 0:
                existing_sign = np.sign(self.particles_data[0][1])
                if current_sign != existing_sign:
                    raise ValueError("Les particules doivent avoir des charges de même signe.")

            # Arrondir légèrement pour la comparaison (éviter pbs de flottants)
            particle_info = (round(mass_u, 5), round(charge_e, 5))
            # Vérifier si la particule (arrondie) existe déjà
            existing_particles = [(round(p[0], 5), round(p[1], 5)) for p in self.particles_data]

            if particle_info not in existing_particles:
                # Ajouter les valeurs originales (non arrondies)
                self.particles_data.append((mass_u, charge_e))
                # Afficher avec formatage dans le Treeview
                self.particle_tree.insert('', tk.END, values=(f"{mass_u:.3f}", f"{charge_e:+.2f}"))
                self.status_var.set(f"{source_info} ajoutée: {mass_u:.3f} u, {charge_e:+.2f} e")
                return True # Ajout réussi
            else:
                # Gérer le cas du doublon (message différent selon la source?)
                if "Raccourci" in source_info:
                    self.status_var.set(f"{source_info} ({mass_u:.3f}u, {charge_e:+.2f}e) déjà présente.")
                else:
                    messagebox.showwarning("Doublon", f"La particule {mass_u:.3f} u / {charge_e:+.2f} e est déjà dans la liste.", parent=getattr(self, 'molecule_fenetre', self.root)) # Parent correct pour msgbox
                self.status_var.set("Ajout annulé (doublon).")
                return False # Ajout échoué (doublon)

        except ValueError as e:
            # Afficher l'erreur dans la bonne fenêtre si possible
            parent_window = getattr(self, 'molecule_fenetre', self.root)
            if parent_window.winfo_exists(): # Vérifier si la fenêtre existe encore
                 messagebox.showerror("Erreur de Validation", f"{e}", parent=parent_window)
            else:
                 messagebox.showerror("Erreur de Validation", f"{e}")
            self.status_var.set(f"Erreur validation: {e}")
            return False # Ajout échoué (validation)
        except Exception as e:
             parent_window = getattr(self, 'molecule_fenetre', self.root)
             if parent_window.winfo_exists():
                 messagebox.showerror("Erreur Inattendue", f"{e}", parent=parent_window)
             else:
                  messagebox.showerror("Erreur Inattendue", f"{e}")
             self.status_var.set("Erreur interne lors de l'ajout.")
             return False # Ajout échoué (autre erreur)


    def remove_particle(self):
        """Supprime la ou les particules sélectionnées dans le Treeview."""
        selected_items = self.particle_tree.selection()
        if not selected_items:
            messagebox.showwarning("Aucune Sélection", "Veuillez sélectionner une ou plusieurs particules à supprimer dans la liste.")
            return

        # Confirmation pour suppression multiple
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
            except tk.TclError:
                print(f"Avertissement: Impossible de trouver l'index pour l'item {item_id} (peut-être déjà supprimé)")

        # Supprimer de la liste de données (en partant de la fin)
        indices_to_remove.sort(reverse=True)
        deleted_count = 0
        for index in indices_to_remove:
            try:
                del self.particles_data[index]
                deleted_count += 1
            except IndexError:
                 print(f"Avertissement: Index {index} hors limites pour self.particles_data.")


        # Supprimer du Treeview
        for item_id in items_to_remove_tree:
            if self.particle_tree.exists(item_id):
                self.particle_tree.delete(item_id)

        self.status_var.set(f"{deleted_count} particule(s) supprimée(s).")

    # --- Widgets Magnétiques ---
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

    # --- Callbacks Sliders Magnétiques ---
    def _on_bz_slider_change(self, event=None):
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
        self.v0_label_var.set(f"{self.v0_var.get():.2e} m/s")

    # --- Widgets Électriques ---
    def create_electric_widgets(self, parent):
        """
        Crée les widgets pour la simulation de déviation électrique, y compris l'incertitude.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)


        # Frames pour afficher/cacher
        self.dynamic_electric_inputs_frame = ttk.Frame(frame)
        self.base_electric_inputs_frame = ttk.Frame(frame)

        # Widgets Communs
        self.angle_var = tk.StringVar(value="30")
        self.add_labeled_entry(frame, "Angle Initial (° vs +y):", self.angle_var).pack(fill=tk.X, pady=3)
        self.dist_var = tk.StringVar(value="0.05")
        self.add_labeled_entry(frame, "Distance/Hauteur (m):", self.dist_var).pack(fill=tk.X, pady=3)
        
        # Checkbox pour choisir le mode
        self.dynamic_elec_var = tk.BooleanVar(value=False)
        dynamic_elec_check = ttk.Checkbutton(frame, text="Mode Dynamique (Sliders)",
                                             variable=self.dynamic_elec_var, command=self.toggle_dynamic_electric)
        dynamic_elec_check.pack(anchor=tk.W, pady=5)

        # --- Widgets Mode Non-Dynamique (Statique) ---
        parent_base = self.base_electric_inputs_frame
        self.v0_elec_var = tk.StringVar(value="1e5") # m/s
        self.add_labeled_entry(parent_base, "Vitesse Initiale (m/s):", self.v0_elec_var).pack(fill=tk.X, pady=3)
        self.diff_pot_var = tk.StringVar(value="-5000") # Volts
        self.add_labeled_entry(parent_base, "Diff. Potentiel (V):", self.diff_pot_var).pack(fill=tk.X, pady=3)
        # Bouton pour lancer la simulation statique
        trace_btn_base = ttk.Button(parent_base, text="Tracer Simulation", command=self.run_electric_simulation)
        trace_btn_base.pack(pady=15)

        # --- Widgets Mode Dynamique ---
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
        self.v0_slider_elec = ttk.Scale(self.slider_frame_v0_elec, from_=1e4, to=2e5, orient=tk.HORIZONTAL,
                                        variable=self.v0_var_elec, command=self._on_v0_slider_change_elec)
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
        self.pot_slider = ttk.Scale(self.slider_frame_v, from_=-10000, to=10000, orient=tk.HORIZONTAL,
                                    variable=self.pot_var, command=self._on_pot_slider_change)
        self.pot_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pot_label_var = tk.StringVar(value=f"{self.pot_var.get():.1f} V")
        ttk.Label(self.slider_frame_v, textvariable=self.pot_label_var, width=12).pack(side=tk.LEFT)
        
        # --- Section Incertitude ---
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)
        self.show_uncertainty_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Afficher Incertitudes (selon deviation.py)", variable=self.show_uncertainty_var, command=self.toggle_uncertainty_inputs).pack(anchor=tk.W, padx=5, pady=(0, 5))
        self.uncertainty_inputs_frame = ttk.LabelFrame(frame, text="Paramètres d'incertitude (%)")
        self.delta_v0_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔV0 (%):", self.delta_v0_percent_var).pack(fill=tk.X, pady=2, padx=5)
        self.delta_theta_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "Δθ (% de l'angle):", self.delta_theta_percent_var).pack(fill=tk.X, pady=2, padx=5) # Note: % angle
        self.delta_h_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔHauteur (%):", self.delta_h_percent_var).pack(fill=tk.X, pady=2, padx=5) # Renommé 'h' pour correspondre au dict
        self.delta_E_percent_var = tk.StringVar(value="1.0")
        self.add_labeled_entry(self.uncertainty_inputs_frame, "ΔChamp E (%):", self.delta_E_percent_var).pack(fill=tk.X, pady=2, padx=5) # % E directement
        ttk.Label(self.uncertainty_inputs_frame, text="Note: Δm/m et Δq/q utiliseront des valeurs fixes (ex: 0.1%)", font=('Segoe UI', 8)).pack(pady=(5,0))
        
        # Bouton pour appliquer les limites et tracer initialement
        apply_limits_btn_dyn = ttk.Button(parent_dyn, text="Appliquer Limites & Tracer", command=self.run_electric_simulation)
        apply_limits_btn_dyn.pack(pady=15)

        # Afficher/Cacher initialement
        self.toggle_dynamic_electric()
        self.toggle_uncertainty_inputs()

    def toggle_dynamic_electric(self):
        """Gère l'affichage des widgets électriques dynamique/statique."""
        is_dynamic = self.dynamic_elec_var.get()
        before_widget = self.uncertainty_inputs_frame if self.show_uncertainty_var.get() and self.uncertainty_inputs_frame.winfo_ismapped() else None
        if is_dynamic:
            self.base_electric_inputs_frame.pack_forget(); self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, before=before_widget)
        else:
            self.dynamic_electric_inputs_frame.pack_forget(); self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5, before=before_widget)
        self.root.after(50, self._update_scroll_region_and_bar)

    def toggle_dynamic_electric(self):
        """Gère l'affichage des widgets électriques selon le mode choisi."""
        if self.dynamic_elec_var.get():
            self.base_electric_inputs_frame.pack_forget()
            self.dynamic_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5)
        else:
            self.dynamic_electric_inputs_frame.pack_forget()
            self.base_electric_inputs_frame.pack(fill=tk.X, pady=5, padx=5)

    def toggle_uncertainty_inputs(self):
        """Affiche ou cache le frame des entrées d'incertitude et ajuste le layout."""
        show = self.show_uncertainty_var.get()
        is_dynamic = self.dynamic_elec_var.get()
        before_widget = self.dynamic_electric_inputs_frame if is_dynamic else self.base_electric_inputs_frame
        if show: self.uncertainty_inputs_frame.pack(fill=tk.X, pady=5, padx=5, before=before_widget)
        else: self.uncertainty_inputs_frame.pack_forget()
        self.root.after(50, self._update_scroll_region_and_bar)
    
    # --- Callbacks Sliders Électriques ---
    def _on_pot_slider_change(self, event=None):
        self._update_pot_label()
        if self.particles_data:
            self.run_electric_simulation(called_by_slider=True)

    def _on_v0_slider_change_elec(self, event=None) :
        self._update_v0_label_elec()
        if self.particles_data:
            self.run_electric_simulation(called_by_slider=True)

    def _update_pot_label(self, event=None):
        self.pot_label_var.set(f"{self.pot_var.get():.1f} V")

    def _update_v0_label_elec(self, event=None):
        self.v0_label_var_elec.set(f"{self.v0_var_elec.get():.2e} m/s")

    # --- Helper ---
    def add_labeled_entry(self, parent, label_text, string_var):
        """Crée une paire Label + Entry."""
        entry_frame = ttk.Frame(parent)
        # Augmenter la largeur du label pour l'alignement
        ttk.Label(entry_frame, text=label_text, width=22, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        entry = ttk.Entry(entry_frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return entry_frame

    # --- Exécution des Simulations ---
    def run_magnetic_simulation(self, called_by_slider=False):
        """Lance la simulation de déviation magnétique."""
        if not self.particles_data:
            if not called_by_slider:
                messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla()
            self.ax.set_title("Déviation Magnétique")
            self.ax.set_xlabel("Position x (m)")
            self.ax.set_ylabel("Position y (m)")
            self.canvas.draw()
            return

        try:
            # Lire X détecteur (commun)
            x_detecteur_str = self.x_detecteur_var.get().strip().replace(',', '.')
            if not x_detecteur_str: raise ValueError("X détecteur ne peut être vide.")
            x_detecteur = float(x_detecteur_str)
            if x_detecteur <= 0: raise ValueError("X détecteur doit être positif.")

            # Lire V0 et Bz selon le mode
            if not self.dynamic_trace_var.get():
                # Mode Statique
                v0_str = self.v0_mag_var.get().strip().replace(',', '.')
                bz_str = self.bz_mag_var.get().strip().replace(',', '.')
                if not v0_str or not bz_str: raise ValueError("V0 et Bz ne peuvent être vides.")
                v0 = float(v0_str)
                bz = float(bz_str)
                if v0 <= 0: raise ValueError("Vitesse initiale doit être > 0.")
                if bz == 0: raise ValueError("Champ magnétique ne peut être nul.")

            else:
                # Mode Dynamique
                # Si appelé par bouton, mettre à jour les limites/valeurs
                if not called_by_slider:
                    bz_min_str = self.bz_min_var.get().strip().replace(',', '.')
                    bz_max_str = self.bz_max_var.get().strip().replace(',', '.')
                    v0_min_str = self.v0_min_var.get().strip().replace(',', '.')
                    v0_max_str = self.v0_max_var.get().strip().replace(',', '.')
                    if not all([bz_min_str, bz_max_str, v0_min_str, v0_max_str]):
                         raise ValueError("Les limites min/max ne peuvent être vides.")

                    bz_min = float(bz_min_str)
                    bz_max = float(bz_max_str)
                    v0_min = float(v0_min_str)
                    v0_max = float(v0_max_str)

                    # Validation des limites
                    if v0_min <= 0 : raise ValueError("V0 min doit être > 0.")
                    if v0_min >= v0_max : raise ValueError("V0 max doit être > V0 min.")
                    if bz_min >= bz_max : raise ValueError("Bz max doit être > Bz min.")

                    # Préserver la valeur des sliders si possible
                    current_bz = self.bz_var.get()
                    current_v0 = self.v0_var.get()

                    self.bz_slider.config(from_=bz_min, to=bz_max)
                    if not (bz_min <= current_bz <= bz_max):
                        bz_init = (bz_max + bz_min) / 2
                        # Éviter de mettre 0 si ce n'est pas explicitement la limite min/max
                        if abs(bz_init) < 1e-9 and (bz_min != 0 or bz_max != 0):
                            bz_init = bz_min if abs(bz_min) > 1e-9 else bz_max # Choisir une limite non nulle
                        self.bz_var.set(bz_init)
                    self._update_bz_label()

                    self.v0_slider.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max):
                        v0_init = (v0_max + v0_min) / 2
                        self.v0_var.set(v0_init)
                    self._update_v0_label()

                # Lire la valeur actuelle des sliders
                v0 = self.v0_var.get()
                bz = self.bz_var.get()
                if abs(bz) < 1e-15: 
                     raise ValueError("Le champ magnétique (Bz) est trop proche de zéro.")

            # --- Exécution Tracé ---
            self.ax.cla() 
            self.status_var.set("Calcul déviation magnétique...")
            self.root.update_idletasks()

            partie_electroaimant.tracer_ensemble_trajectoires(
                    self.particles_data, v0, bz, x_detecteur, create_plot=False, ax=self.ax
                )

            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            self.canvas.draw()
            self.status_var.set("Tracé déviation magnétique terminé.")

        except ValueError as e:
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Magnétique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Mag): {e}")
        except Exception as e:
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Une erreur est survenue (Magnétique):\n{e}", parent=self.root)
            print(f"Erreur Simulation Magnétique: {type(e).__name__}: {e}")
            self.status_var.set("Erreur simulation magnétique.")

    def run_electric_simulation(self, called_by_slider=False):
        """Lance la simulation de déviation électrique en utilisant le nouveau module deviation.py."""
        if not self.particles_data:
            if not called_by_slider: messagebox.showwarning("Aucune Particule", "Veuillez ajouter au moins une particule.", parent=self.root)
            self.status_var.set("Ajoutez des particules pour simuler.")
            self.ax.cla()
            self.ax.set_title("Déviation Électrique")
            self.ax.set_xlabel("x (m)")
            self.ax.set_ylabel("y (m)")
            self.canvas.draw(); return

        try:
            # Paramètres communs
            angle_deg_str = self.angle_var.get().strip().replace(',', '.')
            dist_str = self.dist_var.get().strip().replace(',', '.')
            if not angle_deg_str or not dist_str: raise ValueError("Angle et Distance/Hauteur requis.")
            angle_deg = float(angle_deg_str)
            hauteur_distance = float(dist_str)

            if hauteur_distance <= 0 : raise ValueError("Hauteur/Distance doit être > 0.")
            if not (0 < angle_deg < 90): raise ValueError("Angle doit être > 0° et < 90°.")
            angle_rad = np.radians(angle_deg)
            hauteur_initiale = hauteur_distance

            # Lire V0 et Potentiel selon le mode
            if not self.dynamic_elec_var.get(): 
                # Mode Statique
                v0_str = self.v0_elec_var.get().strip().replace(',', '.')
                pot_str = self.diff_pot_var.get().strip().replace(',', '.')
                if not v0_str or not pot_str: raise ValueError("V0 et Potentiel requis.")
                v0 = float(v0_str)
                potentiel = float(pot_str)
                if v0 <= 0 : raise ValueError("Vitesse initiale doit être > 0.")

            else :
                # Mode Dynamique
                if not called_by_slider :
                    # Mise à jour limites/valeurs
                    pot_min_str = self.diff_pot_min_var.get().strip().replace(',', '.')
                    pot_max_str = self.diff_pot_max_var.get().strip().replace(',', '.')
                    v0_min_str = self.elec_v0_min_var.get().strip().replace(',', '.')
                    v0_max_str = self.elec_v0_max_var.get().strip().replace(',', '.')
                    if not all([pot_min_str, pot_max_str, v0_min_str, v0_max_str]):
                         raise ValueError("Les limites min/max ne peuvent être vides.")

                    pot_min = float(pot_min_str)
                    pot_max = float(pot_max_str)
                    v0_min = float(v0_min_str)
                    v0_max = float(v0_max_str)

                    if v0_min <= 0 : raise ValueError("V0 min doit être > 0.")
                    if v0_min >= v0_max : raise ValueError("V0 max doit être > V0 min.")
                    if pot_min >= pot_max : raise ValueError("Potentiel max doit être > Potentiel min.")

                    current_pot = self.pot_var.get()
                    current_v0 = self.v0_var_elec.get()

                    self.pot_slider.config(from_=pot_min, to=pot_max)
                    if not (pot_min <= current_pot <= pot_max):
                        pot_init = (pot_max + pot_min) / 2
                        self.pot_var.set(pot_init)
                    self._update_pot_label()

                    self.v0_slider_elec.config(from_=v0_min, to=v0_max)
                    if not (v0_min <= current_v0 <= v0_max):
                        v0_init = (v0_max + v0_min) / 2
                        self.v0_var_elec.set(v0_init)
                    self._update_v0_label_elec()

                v0 = self.v0_var_elec.get()
                potentiel = self.pot_var.get()
            
            
            masse_charge_list = self.particles_data

            # --- Effacer et Lancer Simulation ---
            self.ax.cla()
            show_uncertainty = self.show_uncertainty_var.get()
            self.status_var.set(f"Calcul déviation électrique {'avec' if show_uncertainty else 'sans'} incertitude...")
            self.root.update_idletasks()

            if show_uncertainty:
                # Préparer le dictionnaire d'incertitudes (en fractions)
                try:
                    incertitudes_dict = {
                        'v0': float(self.delta_v0_percent_var.get().strip().replace(',', '.')) / 100.0,
                        'theta': float(self.delta_theta_percent_var.get().strip().replace(',', '.')) / 100.0, # % de l'angle
                        'h': float(self.delta_h_percent_var.get().strip().replace(',', '.')) / 100.0, # % hauteur/distance
                        'E': float(self.delta_E_percent_var.get().strip().replace(',', '.')) / 100.0, # % Champ E
                        'm': 0.001, # 0.1% fixe
                        'q': 0.0001 # 0.01% fixe
                    }
                    # Appeler la fonction avec incertitudes
                    deviation.tracer_ensemble_trajectoires_avec_incertitudes(
                        masse_charge_list,
                        vitesse_initiale=v0,
                        incertitudes=incertitudes_dict,
                        potentiel=potentiel,
                        angle_initial=angle_rad,
                        hauteur_initiale=hauteur_initiale,
                        create_plot=False,
                        ax=self.ax
                    )
                    self.status_var.set("Tracé électrique avec incertitudes terminé.")

                except ValueError as e_inc:
                     messagebox.showerror("Erreur Incertitude", f"Valeur d'incertitude invalide:\n{e_inc}", parent=self.root)
                     self.status_var.set("Erreur paramètre incertitude.")
                     # Optionnel: tracer sans incertitude en cas d'erreur ?
                     # deviation.tracer_ensemble_trajectoires(...)
                     # self.canvas.draw()
                     return # Arrêter ici si erreur incertitude
                except KeyError as e_key:
                    messagebox.showerror("Erreur Code", f"Clé manquante dans le dict incertitudes (interne):\n{e_key}", parent=self.root)
                    self.status_var.set("Erreur interne incertitude.")
                    return

            else:
                # Appeler la fonction sans incertitudes
                deviation.tracer_ensemble_trajectoires(
                    masse_charge_list,
                    vitesse_initiale=v0,
                    potentiel=potentiel,
                    angle_initial=angle_rad,
                    hauteur_initiale=hauteur_initiale,
                    create_plot=False,
                    ax=self.ax
                )
                self.status_var.set("Tracé électrique terminé.")

            # --- Finalisation du plot ---
            # Le module deviation devrait gérer la légende et les limites
            # On redessine juste le canvas
            self.canvas.draw()
            self.status_var.set("Tracé déviation électrique terminé.")

        except ValueError as e:
            # Gérer les erreurs de paramètres (V0, angle, hauteur, charges...)
            if not called_by_slider: messagebox.showerror("Erreur Paramètre", f"Paramètre invalide (Électrique):\n{e}", parent=self.root)
            self.status_var.set(f"Erreur paramètre (Elec): {e}")
        except Exception as e:
            # Gérer les erreurs inattendues de la simulation
            if not called_by_slider: messagebox.showerror("Erreur Simulation", f"Erreur inattendue (Électrique):\n{e}", parent=self.root)
            print(f"Erreur Simulation Électrique: {type(e).__name__}: {e}")
            self.status_var.set("Erreur simulation électrique.")


if __name__ == "__main__":
    # Création et lancement de l'application Tkinter
    root = tk.Tk()
    app = ParticleApp(root)
    root.mainloop()