# Champ électrique

L'expression de la force de Lorentz est donnée par :
$
\vec{f} = q(\vec{E} + \vec{v}\wedge\vec{B}),
$
avec $\vec{E}$ le champ magnétique, $\vec{B}$ le champ magnétique, et q la charge de la particule que l'on considère.  

Ici, on suppose un champ électrique constant dans l'espace, et un champ magnétique nul.  
Ainsi, on a : 
$$    
\boldsymbol{\vec{f} = \vec{F}_E = q\vec{E}}
$$

On applique ensuite le Principe Fondamental de la Dynamique à notre particule.
$$
\sum\vec{F}_{ext} = m \vec{a}  
\Rightarrow q\vec{E} = m \vec{a}$$
$$
\Rightarrow \vec{E} = \frac{m}{q} \, \vec{a}
$$
avec $\vec{a}$ l'accélération de la particule et $m$ sa masse.  
En exprimant nos vecteurs dans la base cartésienne $(\vec{u_x}, \vec{u_y}, \vec{u_z})$, on a : 
$$
\begin{pmatrix}
E_x \\
E_y \\
E_z
\end{pmatrix}
=
\frac{m}{q}
\begin{pmatrix}
a_x \\
a_y \\
a_z
\end{pmatrix}
$$
Cette équation vectorielle se traduit dans le système suivant : 
$$
\begin{cases}
E_x = \frac{m}{q} \, a_x\\ 
E_y = \frac{m}{q} \, a_y\\
E_z = \frac{m}{q} \, a_z
\end{cases}
\iff 
\begin{cases}
a_x = \frac{q}{m} \, E_x \\
a_y = \frac{q}{m} \, E_y \\
a_z = \frac{q}{m} \, E_z
\end{cases}
$$

En intégrant le système par rapport au temps t, on a :

$$
\begin{cases}
v_x(t) = \frac{q}{m} E_x t + C_1 \\
v_y(t) = \frac{q}{m} E_y t + C_2 \\
v_z(t) = \frac{q}{m} E_z t + C_3
\end{cases}
$$
avec $ C_1, C_2 $ et $ C_3 $ des constantes à déterminer. Pour cela, on utilise les conditions initiales du problème. En t = 0, on sait que 
$
\vec{v}(0) = \begin{pmatrix} 
0 \\
v_0 \\
0
\end{pmatrix} 
$.  
En remplaçant : 
$$ 
\begin{cases}
\frac{q}{m} E_x * 0 + C_1 = 0 \\
\frac{q}{m} E_y * 0 + C_2 = v_0 \\ 
\frac{q}{m} E_z * 0 + C_3 = 0
\end{cases}
\Rightarrow
\begin{cases}
C_1 = 0 \\
C_2 = v_0 \\
C_3 = 0
\end{cases}
$$
On a donc : 
$
\begin{cases}
\boldsymbol{v_x(t) = \frac{q}{m} E_x t} \\
\boldsymbol{v_y(t) = \frac{q}{m} E_y t + v_0} \\
\boldsymbol{v_z(t) = \frac{q}{m} E_z t}
\end{cases}
$
.

Enfin, pour avoir les équations de la positon, on intègre une deuxième fois notre sytème par rapport au temps :

$$
\begin{cases}
x(t) = \frac{q}{2m} E_x t^2 + C_1' \\
y(t) = \frac{q}{2m} E_y t^2 + v_0 + C_2' \\
z(t) = \frac{q}{2m} E_z t^2 + C_3'
\end{cases}
$$
avec $ C_1, C_2 $ et $ C_3 $ des constantes à déterminer. Pour cela, on utilise les conditions initiales du problème. En t = 0, on sait que :
$$
\begin{pmatrix} 
x \\
y \\
z
\end{pmatrix} 
=
\begin{pmatrix}
0 \\
0 \\
0
\end{pmatrix}
$$  
En remplaçant : 
$$
\begin{cases}
\frac{q}{2m} E_x 0^2 + C_1' = 0 \\
\frac{q}{2m} E_y 0^2 + v_0 + C_2' = 0 \\
\frac{q}{2m} E_z 0^2 + C_3' = 0
\end{cases}
\Rightarrow
\begin{cases}
C_1' = 0 \\
C_2' = 0 \\
C_3' = 0
\end{cases}
$$
Ainsi : 
$
\begin{cases}
\boldsymbol{x(t) = \frac{q}{2m} E_x t^2} \\
\boldsymbol{y(t) = \frac{q}{2m} E_y t^2 + v_0} \\
\boldsymbol{z(t) = \frac{q}{2m} E_z t^2}
\end{cases}
$