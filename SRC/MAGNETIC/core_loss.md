# FICHE RÉCAP : MODÉLISATION MAGNÉTIQUE

## 1. Le Piège du Transformateur : $L_m$ vs $L_k$

Lorsque vous regardez un transformateur de l'extérieur, le courant total mesuré ne reflète pas ce qui se passe dans la ferrite. Il faut diviser le circuit physique en deux éléments théoriques :

* **L'inductance de fuite ($L_k$) :** C'est le flux magnétique qui s'échappe dans l'air. L'air ne sature pas, et il n'y a pas de pertes fer associées. On l'ignore pour calculer l'état de la ferrite.
* **L'inductance magnétisante ($L_m$) :** C'est le flux utile qui traverse la section de ferrite ($A_e$). C'est lui qui risque de saturer le composant et qui crée la chaleur (pertes fer).

**Règle d'or :** La tension qui génère le flux dans le noyau n'est pas la tension d'entrée $v_{in}$, mais la tension interne aux bornes de $L_m$ :

$$v_{Lm}(t) = v_{in}(t) - L_k \frac{di_{pri}(t)}{dt}$$

---

## 2. Calculer le Flux $B(t)$ (Les 2 Méthodes)

### Méthode A : Par le Courant (Idéal pour Inductance / Flyback)

Si vous connaissez le courant magnétisant exact $I_{mag}(t)$ (qui équivaut au courant total pour une simple inductance), la relation est directe et instantanée :

$$B(t) = \frac{L_m \cdot I_{mag}(t)}{N \cdot A_e}$$

### Méthode B : Par la Tension (Idéal pour les Transformateurs DAB / Full-Bridge)

La tension $v_{Lm}(t)$ donne la *vitesse* de variation du flux. En l'intégrant, on obtient l'onde de flux complète.

$$B(t) = B(0) + \frac{1}{N \cdot A_e} \int_{0}^{t} v_{Lm}(\tau) d\tau$$

* **Le Socle $B(0)$ :** C'est la polarisation magnétique continue initiale. Dans un Flyback CCM, on le calcule avec le courant de vallée. Dans un DAB parfaitement symétrique, il est centré automatiquement : $B(0) = -\frac{\Delta B}{2}$.
* **L'Ondulation ($\Delta B$) :** C'est l'aire sous la courbe de la tension positive divisée par $N \cdot A_e$.

---

## 3. Dimensionnement : Saturation et Thermique

Une fois que votre script a généré le tableau complet $B(t)$, vous devez en extraire deux informations cruciales pour valider le composant magnétique.

### 🔴 Vérifier la Saturation ($B_{max}$)

Vous devez extraire le pic absolu pour vérifier que le noyau ne s'effondre pas magnétiquement.

$$B_{max} = \max(B(t))$$

* **Critère :** Doit rester inférieur aux spécifications de la datasheet (généralement $< 0.3\text{ T}$ pour une ferrite de puissance à 100°C). Si vous dépassez cette limite, la perméabilité s'effondre, le courant s'envole de manière incontrôlable, et le pont de puissance grille.

### 🌡️ Calculer les Pertes Fer (Core Losses via iGSE)

Pour estimer l'échauffement de la ferrite sous une forme d'onde complexe (carré, triangle, présence de temps morts), on utilise l'excursion totale ($\Delta B$) et la vitesse instantanée de variation du flux ($\frac{dB}{dt}$).

$$\Delta B = \max(B(t)) - \min(B(t))$$

$$P_{volumique} = \frac{1}{T} \int_{0}^{T} k_i \left\vert{} \frac{v_{Lm}(t)}{N \cdot A_e} \right\vert{}^\alpha (\Delta B)^{\beta - \alpha} dt$$

* **Avantage iGSE :** Contrairement à l'équation de Steinmetz classique (SSE) qui suppose un signal purement sinusoïdal, l'iGSE "punit" thermiquement les instants où la tension est très forte (et donc où le flux varie très violemment), ce qui correspond exactement aux contraintes du découpage en électronique de puissance.