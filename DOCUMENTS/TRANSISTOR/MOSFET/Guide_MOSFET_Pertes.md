# Modèle de pertes MOSFET — Guide complet

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Théorie — mécanismes de pertes dans un MOSFET de puissance](#2-theorie)
3. [Modèle thermique](#3-modele-thermique)
4. [Architecture du module](#4-architecture)
5. [GateDriver — circuit de commande de grille](#5-gatedriver)
6. [SwitchOperatingPoint — point de fonctionnement](#6-switchoperatingpoint)
7. [MOSFET — paramètres composant et utilitaires](#7-mosfet)
8. [Dissipateur thermique externe](#8-dissipateur)
9. [Calcul des pertes — MOSFET solo](#9-mosfet-solo)
10. [Modèle Q_rr — recouvrement inverse de la body diode](#10-modele-qrr)
11. [Mise en parallèle de dies (N > 1)](#11-n-dies-en-parallele)
12. [HalfBridge — demi-pont](#12-halfbridge)
13. [FullBridge — pont complet](#13-fullbridge)
14. [Visualisation — classes Plotter](#14-visualisation-plotters)
15. [Exemples complets](#15-exemples-complets)
16. [Notes et limites du modèle](#16-notes-et-limites)

---

## 1. Introduction

Ce module permet de calculer les **pertes de puissance et la température de jonction** d'un
ou plusieurs MOSFETs de puissance dans un convertisseur DC/DC ou un onduleur. Il couvre les
cas du MOSFET seul, du demi-pont (HalfBridge) et du pont complet (FullBridge).

### Pourquoi ce modèle ?

Le calcul des pertes MOSFET à la main est fastidieux et source d'erreurs pour plusieurs raisons :

- Les pertes par conduction dépendent de R_DS(on), qui lui-même varie avec la température
  de jonction Tj. Tj dépend des pertes. Ce **couplage thermique-électrique** impose une
  résolution itérative qui converge vers l'équilibre.
- Les pertes par commutation dépendent du courant de grille, qui dépend du driver et des
  résistances de grille — elles-mêmes à répartir entre le driver IC, la résistance externe
  et la résistance interne du MOSFET.
- Le **recouvrement inverse Q_rr** de la body diode est une source de pertes souvent
  négligée ou mal calculée : il varie avec le courant conduit et le di/dt, et en bridge,
  il est dissipé dans le switch *opposé* à celui dont la diode se recouvre.
- En cas de mise en parallèle de dies, les scalings sur le driver, les temps de commutation
  et la thermique sont non triviaux.

Ce module automatise tous ces calculs et garantit une allocation correcte des pertes.

### À qui s'adresse ce guide ?

Il s'adresse à un ingénieur électronique de puissance familier avec les convertisseurs
DC/DC et les datasheets MOSFET, mais pas nécessairement avec le code Python.
Aucune connaissance préalable du code source n'est requise.

---

## 2. Théorie — mécanismes de pertes dans un MOSFET de puissance

Un MOSFET de puissance dissipe de l'énergie selon six mécanismes distincts.

### 2.1 Pertes par conduction

Quand le MOSFET est passant, le courant traverse R_DS(on). La puissance dissipée est :

```
P_cond = R_DS(on) × I_rms²
```

R_DS(on) augmente significativement avec la température de jonction selon un modèle linéaire :

```
R_DS(T_j) = R_DS_25 × (1 + α_R × (T_j − 25))
```

`α_R` est typiquement 0.004 à 0.007 K⁻¹ pour un MOSFET Si. Cela signifie qu'à 125 °C,
R_DS(on) peut être 2 à 3 fois plus élevé qu'à 25 °C — et donc les pertes de conduction
2 à 9 fois plus importantes.

### 2.2 Pertes par commutation (croisement V·I)

Durant les transitions, le MOSFET passe par un état intermédiaire où **V_DS et I_D sont
simultanément élevés**. L'énergie dissipée à chaque transition est modélisée par un triangle :

```
E_on  = ½ × V_turn_on  × I_turn_on  × t_on    [J]
E_off = ½ × V_turn_off × I_turn_off × t_off   [J]
P_sw  = (E_on + E_off) × f_sw                  [W]
```

Les durées `t_on` et `t_off` dépendent de la vitesse à laquelle la grille peut être chargée
(courant de grille) et de la charge de commutation Q_sw du composant (section 10 pour le détail).
Plus f_sw est élevée, plus ces pertes dominent.

### 2.3 Pertes de grille

Chaque cycle, l'énergie `Q_g × (V_on − V_off)` est injectée dans la grille pour allumer
le MOSFET, puis extraite pour l'éteindre. Cette énergie est intégralement dissipée dans les
résistances de la boucle de grille, répartie au prorata de chaque résistance :

```
P_gate_total = Q_g × (V_on − V_off) × f_sw   [W]

P_R_g_int  = P_gate_total × R_g_int  / R_loop    (chauffe le die)
P_R_g_ext  = P_gate_total × R_g_ext  / R_loop    (chauffe la résistance externe)
P_driver   = P_gate_total × R_driver / R_loop    (chauffe le driver IC)
```

Seule la part dans `R_g_int` contribue à l'échauffement du MOSFET lui-même.

### 2.4 Pertes C_oss (capacité de sortie)

Au turn-on, le MOSFET court-circuite sa propre capacité de sortie C_oss qui était chargée
à V_bus. L'énergie stockée dans C_oss est dissipée dans le canal du MOSFET :

```
P_coss = ½ × C_oss_eq × V_turn_on² × f_sw   [W]
```

C_oss est très non-linéaire (elle peut varier d'un facteur 10 entre 0 V et V_DSS). Le modèle
utilise la **capacité équivalente en énergie** Co(er), calculée par intégration numérique
de la courbe C_oss(V_DS) fournie dans la datasheet. Si cette courbe n'est pas renseignée,
un fallback sur Q_oss/V_test est utilisé (moins précis, un avertissement est alors émis).

### 2.5 Pertes de conduction de la diode de body

Durant le temps mort (dead-time) en bridge, ou en roue libre dans certains convertisseurs,
le courant passe à travers la diode de body du MOSFET. Cette diode a une tension de seuil
V_F typiquement de 0.7 à 1.2 V, engendrant des pertes :

```
P_diode = V_F_body × I_body_avg × D_body   [W]
```

où `D_body` est le rapport cyclique de conduction de la diode (typiquement `dead_time × f_sw`
pour un bridge).

### 2.6 Pertes par recouvrement inverse Q_rr

Quand une diode passe de l'état conducteur à l'état bloquant, les **porteurs minoritaires
stockés dans la jonction** doivent être évacués avant que la diode ne bloque. Cela génère
un courant inverse transitoire (voir section 10 pour le détail du modèle).

```
P_rr = Q_rr × V_rr × f_sw   [W]
```

Ce phénomène est particulièrement important en bridge : c'est le switch qui s'allume
(et non celui dont la diode se recouvre) qui dissipe cette énergie.

---

## 3. Modèle thermique

### 3.1 Réseau de résistances thermiques

La chaleur générée dans le die parcourt un réseau de résistances jusqu'à l'air ambiant.
Le modèle distingue deux chemins :

```
              [ T_junction (Die) ]
                      │
      ┌───────────────┴───────────────┐
      │                               │
[R_thJC_bot]                   [R_thJC_top]
 (vers pad drain)            (vers dessus boîtier)
      │                               │
 [ T_pad_bot ]                 [ T_case_top ]
      │                               │
[R_th_ext_bot]                 [R_th_ext_top]
 (PCB ou radiateur)          (top-cool ou convection)
      │                               │
      └───────────────┬───────────────┘
                      │
                [ T_ambiant ]
```

- `R_thJC_bot` et `R_thJC_top` : résistances internes au boîtier, données par la datasheet.
- `R_th_ext_bot` et `R_th_ext_top` : résistances thermiques externes, modélisées par
  les classes `Dissipator` (section 8).

### 3.2 Équilibre thermique itératif

Pertes et température sont couplées : R_DS(on) augmente avec Tj, ce qui augmente les pertes
de conduction, ce qui monte Tj, etc. Le modèle résout ce couplage par itération :

```
Tj₀ = T_amb
boucle :
    pertes(Tj) → heating_w
    Tj_new = T_amb + heating_w × R_th_total
    si |Tj_new − Tj| < 0.05 K → convergé
    Tj ← Tj_new
```

La convergence est typiquement atteinte en moins de 10 itérations.

---

## 4. Architecture du module

| Fichier | Contenu |
|---|---|
| `src/Core/Components/MOSFET.py` | `GateDriver`, `SwitchOperatingPoint`, `MOSFET`, `MOSFET_LossReport` |
| `src/Core/Components/Dissipator.py` | `StandardDissipator`, `PCBDissipator`, `Dissipator`, `Placement` |
| `src/Core/Components/BRIDGE.py` | `HalfBridge`, `HalfBridgeContext`, `HalfBridgeOperatingPoint`, `HalfBridgeLossReport`, `FullBridge`, `FullBridgeOperatingPoint`, `FullBridgeLossReport` |
| `src/Core/Components/MOSFET_PLOTTER.py` | `MOSFETPlotter`, `HalfBridgePlotter`, `FullBridgePlotter` |
| `Database/Semiconductors/mosfet_library.json` | Bibliothèque de composants |

**Enchaînement typique (MOSFET solo) :**

```python
drv  = GateDriver(V_on=12, R_out_source=2.2)
op   = SwitchOperatingPoint(I_rms=10, V_turn_on=48, I_turn_on=15,
                             V_turn_off=48, I_turn_off=15, f_sw=100e3, driver=drv)
mos  = MOSFET.from_json("Database/Semiconductors/mosfet_library.json", "BSC016N06NS")
pcb  = PCBDissipator(name="pcb", A_cu_total_side_cm2=6, A_pad_mosfet_cm2=1)
Tj, report = mos.solve_thermal_equilibrium(op, T_amb=25, **pcb.to_mosfet_kwargs())
```

---

## 5. GateDriver

Le `GateDriver` modélise le **circuit de commande de grille** : l'IC driver et les résistances
associées. Il détermine le courant disponible pour charger/décharger la grille, ce qui fixe
les durées de commutation et donc les pertes par croisement V·I.

### Circuit de commande

```
V_on ──[R_out_source]──┬──[R_g_ext_on ]──[R_g_int]──┐
                        │                              Gate
V_off──[R_out_sink  ]──┘──[R_g_ext_off]──[R_g_int]──┘
                                                  C_iss
                                                  GND
```

### Paramètres

| Paramètre | Unité | Description |
|---|---|---|
| `V_on` | V | Tension rail haut du driver (ex: 12 V) |
| `V_off` | V | Tension rail bas (0 V ou négatif pour blocage sûr) |
| `R_out_source` | Ω | Résistance de sortie turn-on — **Modèle A résistif** |
| `R_out_sink` | Ω | Résistance de sortie turn-off — **Modèle A résistif** |
| `I_source_peak` | A | Courant crête max turn-on (saturation du driver) |
| `I_sink_peak` | A | Courant crête max turn-off |
| `I_source_source` | A | Courant constant turn-on — **Modèle B source de courant** |
| `I_sink_source` | A | Courant constant turn-off — **Modèle B source de courant** |

### Deux modèles de driver

**Modèle A — résistif** : le courant crête est calculé depuis la tension disponible et la
résistance totale de boucle. C'est le modèle habituel (driver avec résistance de sortie).

```
I_on = min((V_on − V_plateau) / (R_out_source + R_g_ext_on + R_g_int),  I_source_peak)
```

**Modèle B — source de courant idéale** : courant constant imposé par le driver,
indépendamment des résistances. Correspond aux drivers programmables en courant.

```
I_on = I_source_source   (constant)
```

Les deux modèles peuvent être **mixés** : turn-on en Modèle A, turn-off en Modèle B, ou
l'inverse. En revanche, combiner les deux pour **une même direction** lève une `ValueError`.

### Construction

```python
# Modèle A — driver résistif standard
drv = GateDriver(V_on=12, R_out_source=2.2, R_out_sink=1.0)

# Modèle A avec saturation de courant (driver limité en courant crête)
drv = GateDriver(V_on=12, R_out_source=1.0, I_source_peak=4.0)

# Modèle B turn-on, Modèle A turn-off (driver courant au turn-on pour limiter dV/dt)
drv = GateDriver(V_on=12, I_source_source=2.0, R_out_sink=1.5)

# Chargement depuis la bibliothèque JSON
drv = GateDriver.from_json("Database/Drivers/gate_driver_library.json", "UCC27211A")
drv = GateDriver.from_json("...", "UCC27211A", V_on=15.0)   # override après chargement
```

---

## 6. SwitchOperatingPoint

Le `SwitchOperatingPoint` décrit les **conditions électriques et thermiques du switch** au
point de fonctionnement : tensions et courants commutés, courant RMS, fréquence, conditions
de la diode de body, etc.

### Paramètres obligatoires

| Paramètre | Unité | Description |
|---|---|---|
| `I_rms` | A | Courant drain RMS pendant la phase de conduction |
| `V_turn_on` | V | Tension V_DS bloquée juste avant le turn-on |
| `I_turn_on` | A | Courant I_D commuté lors du turn-on |
| `V_turn_off` | V | Tension V_DS bloquée juste après le turn-off |
| `I_turn_off` | A | Courant I_D commuté lors du turn-off |
| `f_sw` | Hz | Fréquence de commutation |

Pour un buck : `V_turn_on = V_turn_off = V_in`, `I_turn_on ≈ I_turn_off ≈ I_load`.

### Paramètres optionnels

| Paramètre | Défaut | Description |
|---|---|---|
| `driver` | GateDriver() | Circuit de commande (driver IC + résistances) |
| `R_g_ext_on` | 0 Ω | Résistance de grille externe, chemin turn-on |
| `R_g_ext_off` | 0 Ω | Résistance de grille externe, chemin turn-off |
| `T_j` | 25 °C | Température de jonction initiale (mise à jour par solve_thermal_equilibrium) |
| `I_body_avg` | 0 A | Courant moyen dans la diode de body pendant la roue libre |
| `D_body` | 0 | Rapport cyclique de conduction de la diode de body |
| `V_rr` | 0 V | Tension inverse sur la diode au moment du recouvrement |
| `Q_rr_diode_external` | 0 C | Charge Q_rr d'une diode externe (voir section 10) |
| `C_oss_parasite` | None F | Capacité parasite externe au nœud drain-source (PCB, snubber) |
| `t_on` / `t_off` | None | Durées de commutation forcées (None = estimées automatiquement) |

### Estimation automatique de t_on / t_off

Si `t_on` et `t_off` ne sont pas renseignés, le modèle les estime depuis la charge de
commutation Q_sw et le courant de grille :

```
t_on  = Q_sw / I_grille_turn_on
t_off = Q_sw / I_grille_turn_off
```

Le courant de grille est calculé selon le modèle du `GateDriver` (A ou B). Pour utiliser
des valeurs mesurées à l'oscilloscope ou issues de simulation :

```python
op = SwitchOperatingPoint(..., t_on=50e-9, t_off=30e-9)
```

### Paramètres diode de body — conventions importantes

`I_body_avg` et `D_body` décrivent la conduction de la diode de body (roue libre).
`D_body` joue également un rôle clé pour le calcul du Q_rr en bridge :

- **`D_body = 0`** : la diode ne conduit jamais. Le bridge n'attribuera aucune perte Q_rr
  à ce switch (son courant de recouvrement est nul). C'est le cas du HS dans un buck
  **unidirectionnel** où le courant ne remonte jamais.
- **`D_body > 0`** : la diode conduit pendant le temps mort. Le bridge calcule le Q_rr
  de cette body diode automatiquement.

---

## 7. MOSFET — paramètres composant et utilitaires

### Chargement depuis la bibliothèque JSON

```python
mos  = MOSFET.from_json("Database/Semiconductors/mosfet_library.json", "BSC016N06NS")
mos2 = MOSFET.from_json("...", "BSC016N06NS", N=2)   # pack de 2 dies en parallèle
```

Les paramètres présents dans le JSON écrasent les valeurs par défaut. Tout champ absent
du JSON conserve sa valeur par défaut (0 pour les optionnels comme Q_rr_typ).

### Paramètres principaux

| Paramètre JSON | Unité | Description |
|---|---|---|
| `V_DSS` | V | Tension de claquage drain-source |
| `I_max` | A | Courant drain max (informatif) |
| `R_DS_on_25` | Ω | R_DS(on) à 25 °C |
| `alpha_R` | 1/K | Coefficient thermique de R_DS(on) |
| `Q_g` | C | Charge totale de grille (0 V → V_on) |
| `Q_sw` | C | Charge de commutation = Q_gd + Q_gs2 (durée des transitions V/I) |
| `Q_oss` | C | Charge de sortie datasheet, à V_test_oss |
| `V_test_oss` | V | Tension de test associée à Q_oss |
| `V_plateau` | V | Tension de grille lors du plateau de Miller |
| `R_g_int` | Ω | Résistance de grille interne (bonding wire + polysilicium) |
| `V_F_body` | V | Tension de seuil de la diode de body |
| `package` | — | Boîtier de cette entrée (ex: SuperSO8, TOLL, TO-247) |
| `packages_available` | — | Liste de tous les boîtiers disponibles pour ce composant |
| `cost` | € | Coût indicatif unitaire |
| `R_thJC_bot` | K/W | Résistance thermique jonction → pad inférieur (drain) |
| `R_thJC_top` | K/W | Résistance thermique jonction → dessus du boîtier |
| `R_thJA` | K/W + str | R_th jonction-ambiant datasheet + conditions de test |
| `T_j_max` | °C | Température de jonction maximale absolue |

### Paramètres Q_rr (recouvrement inverse — section 10)

| Paramètre JSON | Description |
|---|---|
| `Q_rr_typ` | Charge typique de recouvrement de la body diode [C] |
| `t_rr_typ` | Temps typique de recouvrement [s] |
| `V_R_ref` | Tension inverse du test datasheet [V] |
| `I_F_ref` | Courant direct du test datasheet [A] |
| `di_dt_ref` | di/dt du test datasheet [A/s] |

Mettre `Q_rr_typ = 0` pour les composants SiC/GaN (pas de recouvrement inverse).

### Charges de grille détaillées (optionnelles — modèle di/dt précis)

Ces quatre paramètres se lisent sur la courbe Q_g(Vgs) de la datasheet :

| Paramètre JSON | Unité | Description |
|---|---|---|
| `V_th` | V | Tension de seuil Gate-Source (début de conduction) |
| `Q_gs` | C | Charge Gate-Source totale jusqu'au plateau de Miller |
| `Q_gth` | C | Charge pour atteindre V_th (de 0 V à V_th) |
| `Q_gd` | C | Charge Miller Gate-Drain (largeur du plateau horizontal) |

Quand ces paramètres sont renseignés (`Q_gs > Q_gth > 0`), le calcul du di/dt au turn-on
utilise un **modèle physique fin** basé sur le temps de montée du courant t_ri. Cela améliore
la précision du Q_rr calculé en bridge (voir section 10).

### Courbe C_oss (optionnelle — recommandée)

```json
"vds_points":  [0.1,  5.0, 10.0, 20.0, 30.0, 40.0,  50.0,  60.0],
"coss_points": [4500e-12, 3000e-12, 2100e-12, 1400e-12, 1200e-12, 1100e-12, 1050e-12, 1000e-12]
```

Sans cette courbe, le modèle utilise une approximation linéaire depuis Q_oss/V_test_oss.
Un avertissement est alors émis dans `report.warnings`.

### Utilitaires de calcul

```python
# R_DS(on) corrigé à la température
mos.R_DS_on(T_j=85)           # → Ω

# Facteurs de mérite (comparaison technologique)
mos.fom_rds_qg                 # R_DS × Q_g  [Ω·C]  — qualité gate-oxide globale
mos.fom_rds_qsw                # R_DS × Q_sw [Ω·C]  — compromis conduction / commutation

# Calculer alpha_R depuis deux points R_DS(on) de la datasheet
alpha = MOSFET.calc_alpha_R(R_DS_25=1.6e-3, R_DS_hot=3.2e-3, T_hot=100)

# Calculer Q_sw depuis la courbe de charge de grille
Q_sw = MOSFET.calc_Q_sw(Q_gd=12e-9, Q_gs2=9e-9)
# ou si Q_gs2 n'est pas directement lisible :
Q_sw = MOSFET.calc_Q_sw(Q_gd=12e-9, Q_gs=15e-9, Q_gs1=5e-9)

# Scaler le Q_rr d'une diode externe au point de fonctionnement réel
Q_rr = MOSFET.calc_Q_rr_external(
    Q_rr_ref=30e-9, I_F_ref=10.0, di_dt_ref=100e6,
    I_F=8.0, di_dt=60e6,
)
```

### Capacité de sortie C_oss — calcul en énergie

`C_oss_eq_energy(v_target)` calcule la capacité équivalente en énergie Co(er), c'est-à-dire
la capacité constante fictive qui, chargée à `v_target`, stockerait la même énergie que la
courbe réelle C_oss(V_DS) :

```
E_oss  = ∫₀^v_target v × C_oss(v) dv       (intégration numérique par trapèzes)
Co(er) = 2 × E_oss / v_target²
```

```python
# Courbe renseignée → intégration numérique, précis
c_eq, warning = mos.C_oss_eq_energy(v_target=48.0)   # warning = False

# Courbe absente → fallback Q_oss / V_test, moins précis
c_eq, warning = mos.C_oss_eq_energy(v_target=48.0)   # warning = True

# Accès direct à la capa équivalente en temps (linéaire)
mos.C_oss_eq_time   # = Q_oss / V_test_oss
```

`calculate_detailed_losses()` appelle automatiquement `C_oss_eq_energy()` et ajoute un
avertissement dans `report.warnings` si la courbe est absente.

---

## 8. Dissipateur thermique externe

Le dissipateur modélise le chemin thermique **externe au boîtier** : du pad MOSFET jusqu'à
l'air ambiant. `get_rth()` retourne uniquement R_th externe. R_thJC est ajouté automatiquement
par `solve_thermal_equilibrium()`.

### StandardDissipator

Pour tout radiateur dont la R_th est connue (datasheet constructeur, simulation CFD) :

```python
from src.Core.Components.Dissipator import StandardDissipator, Placement

hs = StandardDissipator(name="heatsink", R_th=3.0, placement=Placement.BOTTOM)

# Avec interface thermique (TIM) en série
hs = StandardDissipator(name="hs_tim", R_th=3.0, R_tim=0.5, placement=Placement.BOTTOM)
```

### PCBDissipator

Modèle thermique analytique d'un plan cuivre PCB. Il modélise deux chemins en parallèle :
- **Chemin A** : convection directe du plan cuivre côté drain vers l'air
- **Chemin B** : conduction à travers le substrat (FR4 + vias thermiques) vers le plan
  cuivre opposé, puis convection

```python
from src.Core.Components.Dissipator import PCBDissipator

# PCB FR4 simple face, sans vias
pcb = PCBDissipator(
    name="pcb_fr4",
    A_cu_total_side_cm2=6.0,    # surface du plan cuivre côté drain [cm²]
    A_pad_mosfet_cm2=0.5,       # surface du pad thermique du MOSFET [cm²]
    pcb_thickness_mm=1.6,
    pcb_k_W_mK=0.3,             # conductivité FR4
)

# PCB FR4 double face avec vias thermiques sous le pad
pcb = PCBDissipator(
    name="pcb_vias",
    A_cu_total_side_cm2=6.0,
    A_cu_other_side_cm2=4.0,    # plan cuivre face opposée [cm²]
    A_pad_mosfet_cm2=0.5,
    n_vias=16,
    via_diameter_mm=0.3,
    via_plating_um=25.0,
    h_conv_eff_W_m2K=12.0,      # convection naturelle + rayonnement ≈ 10–15 W/m²·K
)

# Inspecter le détail du modèle thermique
print(pcb.breakdown())
# → dict : R_conv_side, R_pcb_thru, R_vias, R_path_B, R_total, ...
```

Ordres de grandeur typiques (convection naturelle, 4–6 cm²) :

| Technologie | Vias | R_th approx |
|---|---|---|
| FR4 1.6 mm, sans vias | non | 100–150 K/W |
| FR4 1.6 mm, 16 vias Ø 0.3 mm | oui | 20–30 K/W |
| IMS 0.1 mm diélectrique | – | 5–15 K/W |
| DBC Al₂O₃ 0.38 mm | – | 2–5 K/W |

### Refroidissement double face (dual-side)

Pour un boîtier top-cool ou un module avec deux surfaces de refroidissement :

```python
pcb_bot = PCBDissipator(name="pcb_bot", placement=Placement.BOTTOM, ...)
hs_top  = StandardDissipator(name="hs_top", R_th=5.0, placement=Placement.TOP)

Tj, report = mos.solve_thermal_equilibrium(
    op, T_amb=25,
    **Dissipator.combine_to_mosfet_kwargs(pcb_bot, hs_top)
)
```

---

## 9. Calcul des pertes — MOSFET solo

### Rapport de pertes

`calculate_detailed_losses(op)` retourne un `MOSFET_LossReport` détaillé :

```python
report = mos.calculate_detailed_losses(op)

# Pertes totales du circuit imputées à ce MOSFET
report.total_losses_w           # somme de tout ce qui suit

# Détail par mécanisme
report.conduction_w             # Joule R_DS(on)
report.commutation_w            # croisement V·I au turn-on et turn-off
report.c_oss_w                  # charge/décharge de C_oss
report.gate_internal_w          # pertes dans R_g_int (chauffe le die)
report.gate_external_w          # pertes dans R_g_ext (chauffe la résistance)
report.gate_driver_w            # pertes dans le driver IC
report.diode_body_conduction_w  # chute V_F pendant la roue libre
report.diode_body_rr_w          # recouvrement inverse Q_rr

# Pertes qui échauffent réellement le die
report.heating_self_w           # sans Q_rr ni résistances externes
report.heating_standalone_w     # heating_self + Q_rr (pour MOSFET hors bridge)

# Avertissements éventuels (ex: courbe C_oss absente)
report.warnings
```

### Équilibre thermique itératif

`solve_thermal_equilibrium()` résout le couplage thermique-électrique :

```python
Tj, report = mos.solve_thermal_equilibrium(
    op,
    T_amb=25.0,
    **pcb.to_mosfet_kwargs()    # ou R_th_ext_bot=pcb.get_rth()
)

print(f"Tj = {Tj:.1f} °C  /  Tj_max = {mos.T_j_max:.0f} °C")
print(f"Pertes totales = {report.total_losses_w:.2f} W")
```

La méthode lève `ValueError` si Tj dépasse `T_j_max` (emballement thermique), et
`TimeoutError` si la convergence n'est pas atteinte après 50 itérations (cas pathologique).

---

## 10. Modèle Q_rr — recouvrement inverse de la body diode

### Phénomène physique

Quand une diode conduit dans le sens direct, des porteurs minoritaires s'accumulent dans
la jonction PN. Lorsque la diode est forcée à se bloquer (par l'allumage du switch adverse),
ces porteurs doivent être évacués **avant** que la diode ne soit réellement bloquante.
Pendant ce temps, un courant inverse circule — l'énergie associée est dissipée dans
le circuit.

```
                  ┌─ courant normal ─→
   Diode passante │
                  │
   Extinction     │   ↗ I_rr_peak (courant inverse)
   forcée ────────┼──────────────────────────────── t
                  │    ╲_____________
                  │        t_rr
                  └─ (diode bloque enfin)
```

La charge totale évacuée est Q_rr, et l'énergie dissipée à chaque commutation :

```
P_rr = Q_rr × V_rr × f_sw   [W]
```

Il n'y a pas de facteur ½ : pendant tout le recouvrement, la tension inverse `V_rr` est
déjà établie (elle monte très vite en début d'extinction), donc toute la charge Q_rr est
évacuée sous la pleine tension.

### Modèle de scaling

La datasheet donne `Q_rr_typ` et `t_rr_typ` à un unique point de référence
(`I_F_ref`, `di_dt_ref`, `V_R_ref`). Pour tout autre point de fonctionnement,
le modèle applique un scaling empirique :

```
Q_rr(I_F, di/dt) = Q_rr_typ × √(I_F / I_F_ref) × √(di_dt_ref / di/dt)
t_rr(I_F, di/dt) = t_rr_typ × √(I_F / I_F_ref) × √(di_dt_ref / di/dt)
I_rr_peak        = 2 × Q_rr / t_rr   (approximation triangulaire)
```

Intuition :
- `√(I_F/I_F_ref)` : plus le courant direct est élevé, plus les porteurs stockés sont
  nombreux → Q_rr augmente.
- `√(di_dt_ref/di_dt)` : un di/dt rapide laisse moins de temps à la recombinaison interne
  → plus de charge à extraire de force → Q_rr augmente quand di/dt **diminue**.

```python
Q = mos.scaled_Q_rr(I_F=30, di_dt=50e6)   # [C]
t = mos.scaled_t_rr(I_F=30, di_dt=50e6)   # [s]
I = mos.I_rr_peak(I_F=30, di_dt=50e6)     # [A]
```

### Estimation du di/dt au turn-on

Le di/dt détermine combien de Q_rr est forcé sur la body diode adverse. Il est calculé
automatiquement par `di_dt_turn_on(mos)`. Deux modèles sont disponibles :

**Modèle générique (toujours disponible) :**

```
di/dt = I_turn_on / t_on   avec t_on = Q_sw / I_grille
```

C'est une approximation conservative : `t_on` inclut la descente de V_DS, pas seulement
la montée du courant.

**Modèle physique fin (activé si `V_th`, `Q_gs`, `Q_gth` sont renseignés) :**

La montée du courant drain ne dure que pendant que V_GS monte de V_th au plateau de Miller,
soit une durée t_ri < t_on :

```
I_g_moy = (V_on − (V_th + V_plateau)/2) / Rg_total    (Modèle A)
  ou
I_g_moy = I_source_source                              (Modèle B)

t_ri  = (Q_gs − Q_gth) / I_g_moy
di/dt = I_turn_on / t_ri
```

Le di/dt physique est plus élevé qu'avec le modèle générique (t_ri < t_on), ce qui donne
un Q_rr adverse plus faible — résultat plus réaliste, moins pessimiste.

### Convention d'usage selon le contexte

#### Mode solo — MOSFET seul

`Q_rr_diode_external` dans `SwitchOperatingPoint` représente la charge Q_rr d'une
**diode externe** que ce MOSFET force à s'éteindre en s'allumant.

Par défaut `Q_rr_diode_external = 0` : aucune perte Q_rr n'est modélisée. C'est
l'utilisateur qui décide ce qu'il met dans ce champ. Le `Q_rr_typ` du MOSFET lui-même
n'est pas utilisé en mode solo (il sert au bridge — voir section 12).

**Buck asynchrone (MOSFET + diode de roue libre séparée) :**

```python
# Helper pour scaler le Q_rr de la diode externe au point réel
Q_rr_ext = MOSFET.calc_Q_rr_external(
    Q_rr_ref=30e-9,     # C    — datasheet de la diode, valeur au point de référence
    I_F_ref=10.0,       # A    — courant de test datasheet
    di_dt_ref=100e6,    # A/s  — di/dt de test datasheet
    I_F=8.0,            # A    — courant réel dans la diode juste avant extinction
    di_dt=60e6,         # A/s  — di/dt réel au turn-on du MOSFET
)
op = SwitchOperatingPoint(..., Q_rr_diode_external=Q_rr_ext, V_rr=48.0)
```

Si le point de fonctionnement correspond exactement au point de référence datasheet,
`Q_rr_diode_external=Q_rr_ref` suffit sans appeler le helper.

#### Mode bridge — HalfBridge / FullBridge

En bridge, `Q_rr_diode_external` est **calculé et injecté automatiquement**. L'utilisateur
n'a rien à renseigner pour les body diodes. Voir section 12.

---

## 11. N dies en parallèle

Pour modéliser N dies identiques en parallèle avec un driver partagé :

```python
mos = MOSFET.from_json("...", "BSC016N06NS", N=2)
```

`calculate_detailed_losses()` applique automatiquement les scalings suivants :

| Grandeur | Scaling | Raison |
|---|---|---|
| Conduction | ÷ N | R_DS apparent du pack = R_DS_die / N |
| Commutation V·I | × N | Driver partagé → t_sw allongé (vue d'une grille : R_out×N) |
| C_oss par die, Q_g | × N | Grandeurs propres à chaque die |
| C_oss_parasite | × 1 | Capa externe au pack, indépendante de N |
| Conduction diode de body | × 1 | Courant total inchangé |
| Q_rr_diode_external | × 1 | Charge externe au MOSFET |

**Hypothèse** : driver unique partagé entre les N dies. Si chaque die a son propre driver,
forcer `t_on` et `t_off` manuellement dans `SwitchOperatingPoint`.

**Modèle thermique — deux conventions :**

```python
# shared_dissipator=False [défaut] : N pads séparés, chaque die sur sa propre zone cuivre
# → R_th_ext est la résistance PAR DIE
Tj, report = mos.solve_thermal_equilibrium(
    op, T_amb=25,
    R_th_ext_bot=pcb_par_die.get_rth(),
    shared_dissipator=False)

# shared_dissipator=True : N dies sur un dissipateur commun unique
# → R_th_ext est la résistance TOTALE du dissipateur partagé
Tj, report = mos.solve_thermal_equilibrium(
    op, T_amb=25,
    R_th_ext_bot=heatsink_total.get_rth(),
    shared_dissipator=True)
```

---

## 12. HalfBridge — demi-pont

Le `HalfBridge` modélise un **bras de pont** composé d'un switch High-Side (HS) et d'un
switch Low-Side (LS). C'est la topologie de base du buck synchrone, du boost synchrone,
et d'un bras d'onduleur.

### Allocation correcte des pertes Q_rr

C'est le point le plus délicat d'un bridge : quand le HS s'allume, il force l'extinction
de la body diode du LS. L'énergie `Q_rr × V_bus` est dissipée **dans le die HS**, pas dans
le LS (c'est le HS qui fournit le courant inverse). Le modèle alloue donc :

```
HS s'allume  →  body diode LS se recouvre  →  chaleur dans HS
LS s'allume  →  body diode HS se recouvre  →  chaleur dans LS

heating_high_w = high.heating_self_w + low.diode_body_rr_w
heating_low_w  = low.heating_self_w  + high.diode_body_rr_w
```

Ce couplage est calculé et réalloué **automatiquement**. L'utilisateur définit uniquement
les conditions de fonctionnement de chaque switch.

### Classes utilisées

| Classe | Rôle |
|---|---|
| `HalfBridgeContext` | Conditions communes au bras : V_bus, f_sw, dead_time, T_amb |
| `HalfBridgeOperatingPoint` | Regroupe le contexte + op_high + op_low |
| `HalfBridge` | Calcule les pertes et l'équilibre thermique couplé |
| `HalfBridgeLossReport` | Rapport avec allocation correcte des Q_rr |

### Mise en œuvre — buck synchrone bidirectionnel

```python
from src.Core.Components.BRIDGE import HalfBridge, HalfBridgeContext, HalfBridgeOperatingPoint
from src.Core.Components.MOSFET import GateDriver, SwitchOperatingPoint, MOSFET

# Composants
mos_hs = MOSFET.from_json("...", "BSC016N06NS")
mos_ls = MOSFET.from_json("...", "BSC016N06NS")
drv    = GateDriver(V_on=12, R_out_source=2.2, R_out_sink=1.5)

# Contexte commun
ctx = HalfBridgeContext(V_bus=48, f_sw=100e3, dead_time=100e-9, T_amb=25)

# Buck 48V → 12V, I_out=10A, D=0.25
# Dead-time = 100 ns → D_body = 100e-9 × 100e3 = 0.01
op_high = SwitchOperatingPoint(
    I_rms=10 * (0.25**0.5),   # I_out × √D ≈ 5 A
    V_turn_on=48, I_turn_on=10,
    V_turn_off=48, I_turn_off=10,
    f_sw=100e3,
    I_body_avg=10,    # courant de roue libre pendant le dead-time
    D_body=0.01,      # dead_time × f_sw
    V_rr=48,
    driver=drv,
)
op_low = SwitchOperatingPoint(
    I_rms=10 * ((1-0.25)**0.5),  # I_out × √(1-D) ≈ 8.66 A
    V_turn_on=48, I_turn_on=10,
    V_turn_off=48, I_turn_off=10,
    f_sw=100e3,
    I_body_avg=10,
    D_body=0.01,
    V_rr=48,
    driver=drv,
)

# Assemblage et résolution
op = HalfBridgeOperatingPoint.with_shared_fsw(ctx, op_high, op_low)
hb = HalfBridge(high_side=mos_hs, low_side=mos_ls)

Tj_hs, Tj_ls, report = hb.solve_thermal_equilibrium(
    op,
    R_th_high_ext_bot=5.0,
    R_th_low_ext_bot=5.0,
)
print(f"Tj_HS = {Tj_hs:.1f} °C,  Tj_LS = {Tj_ls:.1f} °C")
print(f"Pertes totales bras = {report.total_bridge_losses_w:.2f} W")
print(f"  HS (avec Q_rr LS) = {report.heating_high_w:.2f} W")
print(f"  LS (avec Q_rr HS) = {report.heating_low_w:.2f} W")
```

### Cas particulier — buck unidirectionnel (body diode HS ne conduit jamais)

Dans un buck unidirectionnel, le courant ne remonte jamais dans le HS : sa body diode
ne conduit jamais et n'accumule aucune charge. Il suffit d'indiquer `D_body=0` dans op_high.
Le bridge détecte `D_body=0` et attribue automatiquement Q_rr_HS = 0 (pas de courant
de recouvrement à calculer).

```python
op_high = SwitchOperatingPoint(
    ...,
    I_body_avg=0.0,   # body diode HS ne conduit pas
    D_body=0.0,       # signal explicite : aucune conduction de diode → Q_rr_HS = 0
    V_rr=0.0,
)
# op_low reste inchangé — la body diode LS conduit pendant le dead-time
```

### Inspection des Q_rr calculés

```python
Q_rr_ls = hb.Q_rr_LS_scaled(op)      # Q_rr body diode LS [C] (forcé par le HS)
Q_rr_hs = hb.Q_rr_HS_scaled(op)      # Q_rr body diode HS [C] (forcé par le LS)
I_peak  = hb.rr_peak_seen_by_HS(op)  # pic de courant inverse vu par le HS [A]
I_peak  = hb.rr_peak_seen_by_LS(op)  # pic de courant inverse vu par le LS [A]
```

---

## 13. FullBridge — pont complet

Le `FullBridge` regroupe deux `HalfBridge` indépendants (jambe A et jambe B). Le couplage
Q_rr reste intra-jambe : HS_A force la body diode de LS_A, mais pas celle de LS_B.

```python
from src.Core.Components.BRIDGE import FullBridge, FullBridgeOperatingPoint

fb    = FullBridge(name="h_bridge", leg_a=hb_a, leg_b=hb_b)
op_fb = FullBridgeOperatingPoint(leg_a=op_hb_a, leg_b=op_hb_b)

Tj_dict, report = fb.solve_thermal_equilibrium(
    op_fb,
    R_th_ext={
        "high_a_bot": 5.0,
        "low_a_bot":  5.0,
        "high_b_bot": 5.0,
        "low_b_bot":  5.0,
    }
)
# Tj_dict → {"high_a": ..., "low_a": ..., "high_b": ..., "low_b": ...}
print(f"Pertes totales pont = {report.total_bridge_losses_w:.2f} W")
```

---

## 14. Visualisation — classes Plotter

`MOSFET_PLOTTER.py` regroupe trois classes de visualisation Plotly réutilisables, une par
topologie. Chaque méthode **retourne** un `go.Figure` (ou un tuple de deux) sans appeler
`.show()` — c'est l'appelant qui contrôle l'affichage, l'export ou la personnalisation.

### Pourquoi des classes Plotter ?

- **Notebooks courts** : les cellules de sweep passent de ~80 lignes à 2–3 lignes.
- **Cohérence visuelle** : mêmes couleurs, même template Plotly, mêmes labels partout.
  Modifier le style dans `MOSFET_PLOTTER.py` propage le changement à tous les notebooks.
- **Figures retournées** : on peut exporter, personnaliser ou chaîner les figures avant de
  les afficher.

```python
fig = plotter.plot_freq_sweep(...)
fig.write_html("rapport.html")         # export HTML interactif
fig.write_image("sweep.png")           # export image
fig.update_layout(title="Mon titre")   # personnalisation avant affichage
fig.show()
```

- **Un seul endroit à maintenir** : corriger une formule ou ajouter une annotation dans la
  classe bénéficie à toutes les utilisations.

---

### MOSFETPlotter — MOSFET solo

```python
from src.Core.Components.MOSFET_PLOTTER import MOSFETPlotter

plotter = MOSFETPlotter(mosfet, op, dissipateur, T_amb, V_bus)
```

| Méthode | Retour | Description |
|---|---|---|
| `plot_coss()` | `go.Figure` | Courbe Coss(Vds) avec Co(tr) et Co(er) au point V_bus |
| `plot_loss_breakdown(T_j, pertes)` | `go.Figure` | Barres de répartition des pertes |
| `plot_freq_sweep(T_j, f_max_khz, n_points)` | `(fig_pertes, fig_Tj)` | Sweep fréquentiel — pertes et Tj estimée |
| `plot_temp_sweep(T_j)` | `go.Figure` | P_cond et R_DS(on) vs Tj |
| `plot_n_parallel(N_list, db_path, diss)` | `(fig_bar, fig_Tj)` | Comparaison N dies en parallèle |

**Exemple — MOSFET solo :**

```python
plotter = MOSFETPlotter(mosfet, op, pcb, T_amb=25, V_bus=48)

# Fiche Coss
plotter.plot_coss().show()

# Répartition des pertes à l'équilibre thermique
Tj, pertes = mosfet.solve_thermal_equilibrium(op, T_amb=25, **pcb.to_mosfet_kwargs())
plotter.plot_loss_breakdown(Tj, pertes).show()

# Sweep fréquentiel 1 kHz → 150 kHz
fig_p, fig_tj = plotter.plot_freq_sweep(Tj, f_max_khz=150)
fig_p.show(); fig_tj.show()

# Sweep thermique (P_cond vs Tj)
plotter.plot_temp_sweep(Tj).show()

# N dies en parallèle : 1, 2, 3, 4
fig_bar, fig_tj = plotter.plot_n_parallel([1, 2, 3, 4], db_path=DB_MOSFET, diss=pcb)
fig_bar.show(); fig_tj.show()
```

---

### HalfBridgePlotter — demi-pont

```python
from src.Core.Components.MOSFET_PLOTTER import HalfBridgePlotter

plotter_hb = HalfBridgePlotter(bridge, op_bridge, diss_hs, diss_ls, T_amb, V_bus)
```

| Méthode | Retour | Description |
|---|---|---|
| `plot_loss_breakdown(Tj_hs, Tj_ls, report)` | `go.Figure` | Barres groupées HS / LS par catégorie de pertes |
| `plot_freq_sweep(Tj_hs, Tj_ls, f_max_khz, n_points)` | `(fig_pertes, fig_Tj)` | Sweep f_sw — pertes individuelles + totaux HS/LS |
| `plot_current_sweep(Tj_hs, Tj_ls, D, I_max, n_points)` | `go.Figure` | Sweep I_load à f_sw et D fixes |

La légende du sweep fréquentiel distingue : Cond HS, Comm HS, C_oss HS, Q_rr→HS (reçu du
LS), Total HS — et de même pour LS. Les totaux sont tracés en `dashdot` plus épais.

**Exemple — HalfBridge :**

```python
plotter_hb = HalfBridgePlotter(bridge, op_bridge, diss_hs, diss_ls, T_amb=25, V_bus=48)

# Répartition des pertes (après solve_thermal_equilibrium)
Tj_hs, Tj_ls, report = bridge.solve_thermal_equilibrium(op_bridge, ...)
plotter_hb.plot_loss_breakdown(Tj_hs, Tj_ls, report).show()

# Sweep fréquentiel
fig_f, fig_tj = plotter_hb.plot_freq_sweep(Tj_hs, Tj_ls, f_max_khz=300)
fig_f.show(); fig_tj.show()

# Sweep courant de charge jusqu'à 40 A, D=0.5
plotter_hb.plot_current_sweep(Tj_hs, Tj_ls, D=0.5, I_max=40).show()
```

---

### FullBridgePlotter — pont complet

```python
from src.Core.Components.MOSFET_PLOTTER import FullBridgePlotter

plotter_fb = FullBridgePlotter(bridge_full, op_full, rth_config, T_amb, V_bus)
```

`rth_config` est un dictionnaire de résistances thermiques effectives (K/W) pour chaque
switch du pont. Les clés `_top` sont optionnelles (top-cool) :

```python
rth_config = {
    "high_a_bot": 5.0,   # R_th_eff HS jambe A (bottom-cool)
    "low_a_bot":  5.0,   # R_th_eff LS jambe A
    "high_b_bot": 5.0,   # R_th_eff HS jambe B
    "low_b_bot":  5.0,   # R_th_eff LS jambe B
    # "high_a_top": ..., # optionnel — top-cool jambe A HS
}
```

| Méthode | Retour | Description |
|---|---|---|
| `plot_loss_breakdown(tjs, report_full)` | `(fig_bar, fig_pie)` | Barres 4 switches + camembert global |
| `plot_freq_sweep(tjs, f_max_khz, n_points)` | `(fig_chaleur, fig_Tj)` | Chaleur dissipée et Tj vs f_sw |
| `plot_d_sweep(tjs, I_a, D_ref, D_range, n_points)` | `go.Figure` | Pertes jambe A vs rapport cyclique D |

`tjs` est un dict `{"high_a": ..., "low_a": ..., "high_b": ..., "low_b": ...}` contenant
les températures de jonction à l'équilibre.

**Exemple — FullBridge :**

```python
plotter_fb = FullBridgePlotter(bridge_full, op_full, rth_config, T_amb=25, V_bus=48)

# Répartition — barres + camembert
fig_bar, fig_pie = plotter_fb.plot_loss_breakdown(tjs, report_full)
fig_bar.show(); fig_pie.show()

# Sweep fréquentiel
fig_q, fig_tj = plotter_fb.plot_freq_sweep(tjs, f_max_khz=200)
fig_q.show(); fig_tj.show()

# Sweep rapport cyclique D (0.1 → 0.9) à courant fixé
plotter_fb.plot_d_sweep(tjs, I_a=20.0, D_ref=0.5).show()
```

Le sweep D affiche en double axe : pertes jambe A (axe gauche) et Tj HS_A / Tj LS_A
(axe droit). Une ligne verticale marque D_ref (point courant) et D_opt (minimum de pertes).

---

## 15. Exemples complets

### Buck asynchrone — MOSFET solo avec Q_rr de la diode de roue libre

```python
from src.Core.Components.MOSFET import GateDriver, SwitchOperatingPoint, MOSFET
from src.Core.Components.Dissipator import PCBDissipator

V_in, V_out, I_out, f_sw = 48.0, 12.0, 10.0, 100e3
D = V_out / V_in   # 0.25

drv = GateDriver(V_on=12, R_out_source=2.2, R_out_sink=1.5)
mos = MOSFET.from_json("Database/Semiconductors/mosfet_library.json", "BSC016N06NS")
pcb = PCBDissipator(
    name="pcb", A_cu_total_side_cm2=6.0, A_pad_mosfet_cm2=0.5,
    pcb_thickness_mm=1.6, pcb_k_W_mK=0.3,
    n_vias=16, via_diameter_mm=0.3, A_cu_other_side_cm2=4.0,
)

# Q_rr de la diode Schottky de roue libre, scalé au point réel
Q_rr_ext = MOSFET.calc_Q_rr_external(
    Q_rr_ref=30e-9, I_F_ref=10.0, di_dt_ref=100e6,
    I_F=I_out, di_dt=80e6,
)

op = SwitchOperatingPoint(
    I_rms=I_out * (D**0.5),
    V_turn_on=V_in, I_turn_on=I_out,
    V_turn_off=V_in, I_turn_off=I_out,
    f_sw=f_sw, driver=drv,
    Q_rr_diode_external=Q_rr_ext,
    V_rr=V_in,
)

Tj, report = mos.solve_thermal_equilibrium(op, T_amb=25, **pcb.to_mosfet_kwargs())

print(f"Tj = {Tj:.1f} °C  /  Tj_max = {mos.T_j_max:.0f} °C")
print(f"Pertes totales        = {report.total_losses_w:.2f} W")
print(f"  Conduction          : {report.conduction_w:.2f} W")
print(f"  Commutation         : {report.commutation_w:.2f} W")
print(f"  C_oss               : {report.c_oss_w:.2f} W")
print(f"  Grille (int/ext/drv): {report.gate_internal_w:.2f} / "
      f"{report.gate_external_w:.2f} / {report.gate_driver_w:.2f} W")
print(f"  Q_rr diode externe  : {report.diode_body_rr_w:.2f} W")
```

### Buck synchrone bidirectionnel — HalfBridge avec N=2 dies côté HS

```python
from src.Core.Components.BRIDGE import HalfBridge, HalfBridgeContext, HalfBridgeOperatingPoint
from src.Core.Components.MOSFET import GateDriver, SwitchOperatingPoint, MOSFET

V_bus, I_out, D, f_sw, t_dt = 48.0, 20.0, 0.25, 100e3, 100e-9
D_body = t_dt * f_sw   # ≈ 0.01

drv    = GateDriver(V_on=12, R_out_source=2.2, R_out_sink=1.5)
mos_hs = MOSFET.from_json("...", "BSC016N06NS", N=2)   # 2 dies en parallèle côté HS
mos_ls = MOSFET.from_json("...", "BSC016N06NS", N=1)

ctx = HalfBridgeContext(V_bus=V_bus, f_sw=f_sw, dead_time=t_dt, T_amb=25)

op_high = SwitchOperatingPoint(
    I_rms=I_out*(D**0.5), V_turn_on=V_bus, I_turn_on=I_out,
    V_turn_off=V_bus, I_turn_off=I_out, f_sw=f_sw,
    I_body_avg=I_out, D_body=D_body, V_rr=V_bus, driver=drv,
)
op_low = SwitchOperatingPoint(
    I_rms=I_out*((1-D)**0.5), V_turn_on=V_bus, I_turn_on=I_out,
    V_turn_off=V_bus, I_turn_off=I_out, f_sw=f_sw,
    I_body_avg=I_out, D_body=D_body, V_rr=V_bus, driver=drv,
)

op = HalfBridgeOperatingPoint.with_shared_fsw(ctx, op_high, op_low)
hb = HalfBridge(high_side=mos_hs, low_side=mos_ls)

Tj_hs, Tj_ls, report = hb.solve_thermal_equilibrium(op,
    R_th_high_ext_bot=3.0,   # dissipateur par die (2 pads séparés)
    R_th_low_ext_bot=5.0,
)
print(f"Q_rr LS (vu par HS) = {hb.Q_rr_LS_scaled(op)*1e9:.1f} nC")
print(f"Tj_HS = {Tj_hs:.1f} °C,  Tj_LS = {Tj_ls:.1f} °C")
```

---

## 16. Notes et limites du modèle

- **Modèle de commutation triangulaire** : `E_on = ½ V I t` suppose des formes d'onde
  linéaires. En réalité les formes dépendent du circuit parasite (inductance de boucle,
  capacités PCB). Pour plus de précision, mesurer `t_on` / `t_off` à l'oscilloscope et
  les forcer dans `SwitchOperatingPoint`.

- **Modèle Q_rr en √** : le scaling `√I × √(1/di_dt)` est une approximation empirique
  (modèle charge stockée junction PIN). Valide pour les MOSFETs Si standard à canal
  modérément dopé. Les MOSFETs SiC/GaN ont un Q_rr très faible ou nul — les modéliser
  avec `Q_rr_typ = 0`.

- **Mise en parallèle** : le modèle suppose un driver unique partagé et un équilibrage
  parfait des courants entre les N dies (même R_DS(on), pas de déséquilibre thermique).
  En pratique, des variations de seuil V_th ou de R_g_int entre dies peuvent causer un
  déséquilibre — à modéliser séparément si critique.
