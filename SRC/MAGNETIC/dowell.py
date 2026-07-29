from enum import Enum
from math import pi, sqrt
from pydantic import BaseModel, Field
import numpy as np

MU0 = 4e-7 * pi
SIGMA_CU_20C = 5.96e7

def skin_depth(freq: float, sigma: float = SIGMA_CU_20C, porosity: float = 1.0) -> float:
    if freq < 1e-3:
        return float('inf')
    omega = 2.0 * pi * freq
    return sqrt(2.0 / (omega * MU0 * sigma * porosity))

def porosity(number_of_turns: int, wire_diameter: float, winding_width: float) -> float:
    return min((number_of_turns * wire_diameter) / winding_width, 1.0)

class WINDING_TYPE(Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"

class POLARITY(Enum):
    POSITIVE = 1
    NEGATIVE = -1

class Layer(BaseModel):
    name: str = ""
    number_of_turns: int 
    bw: float             
    height: float         
    MLT: float            
    winding_type: WINDING_TYPE
    polarity: POLARITY     
    sigma: float = SIGMA_CU_20C 
    porosity: float = 1.0 
    
    def delta(self, freq: float) -> float:
        if freq < 1e-3:
            return 0.0
        return self.height / skin_depth(freq, self.sigma, self.porosity)
    
    def dc_resistance(self) -> float:
        """Vraie résistance DC physique du fil rond"""
        wire_area = pi * (self.height / 2)**2
        return (self.MLT * self.number_of_turns) / (self.sigma * wire_area)
    
    def dc_resistance_dowell(self) -> float:
        """Résistance DC du 'feuillard carré' virtuel vu par Dowell"""
        dowell_area = self.height**2
        return (self.MLT * self.number_of_turns) / (self.sigma * dowell_area)

class Dowell_Winding_Structure(BaseModel):
    list_of_layers: list[Layer] = Field(default_factory=list)
    
    def add_layer(self, layer: Layer):
        self.list_of_layers.append(layer)
        
    def field_at_boundary(self, layer_index: int, currents_rms: dict[WINDING_TYPE, float]) -> float:
        accumulated_ampere_turns = 0.0
        for i in range(layer_index + 1):
            layer = self.list_of_layers[i]
            layer_current = currents_rms.get(layer.winding_type, 0.0)
            accumulated_ampere_turns += layer.number_of_turns * layer_current * layer.polarity.value
            
        return accumulated_ampere_turns / self.list_of_layers[layer_index].bw
    
    def alpha(self, layer_index: int, currents_rms: dict[WINDING_TYPE, float]) -> float:
        if layer_index == 0:
            return 0.0
        h_prev = self.field_at_boundary(layer_index - 1, currents_rms)
        h_curr = self.field_at_boundary(layer_index, currents_rms)
        return h_prev / h_curr if h_curr != 0 else 0.0
   
    def _dowell_G(self, delta: float) -> tuple[float, float]:
        if delta < 1e-3:
            return 1.0, 0.5
        num_g_1 = np.sinh(2 * delta) + np.sin(2 * delta)
        den = np.cosh(2 * delta) - np.cos(2 * delta)
        num_g_2 = np.sinh(delta) * np.cos(delta) + np.cosh(delta) * np.sin(delta)
        
        g1 = (num_g_1 / den) * delta
        g2 = (num_g_2 / den) * delta
        return g1, g2
    
    def loss_at_frequency(self, freq: float, currents_rms: dict[WINDING_TYPE, float]) -> float:
        total_loss = 0.0
        for i, layer in enumerate(self.list_of_layers):
            layer_current = currents_rms.get(layer.winding_type, 0.0)
            if layer_current == 0.0:
                continue
                
            delta_i = layer.delta(freq)
            g1, g2 = self._dowell_G(delta_i)
            
            H_i = self.field_at_boundary(i, currents_rms)
            alpha_i = self.alpha(i, currents_rms)
            
            bracket = (1.0 + alpha_i**2) * g1 - 4.0 * alpha_i * g2
            right_part = H_i**2 * bracket
            left_part = layer.MLT / (layer.height * layer.porosity * layer.sigma)
            
            total_loss += left_part * right_part * layer.bw
            
        return total_loss

    def dc_loss(self, currents_rms: dict[WINDING_TYPE, float]) -> float:
        return sum(
            layer.dc_resistance() * (currents_rms.get(layer.winding_type, 0.0)**2)
            for layer in self.list_of_layers
        )
        
    def physical_losses(self, freq: float, currents_rms: dict[WINDING_TYPE, float]) -> tuple[float, float]:
        """
        Retourne (Total_P_DC, Total_P_AC) avec la correction géométrique pour fil rond.
        """
        total_p_dc_true = 0.0
        total_p_ac_true = 0.0
        
        for i, layer in enumerate(self.list_of_layers):
            layer_current = currents_rms.get(layer.winding_type, 0.0)
            if layer_current == 0.0:
                continue
                
            # --- 1. Pertes DC vraies ---
            r_dc_true = layer.dc_resistance_round()
            p_dc_true = r_dc_true * (layer_current**2)
            total_p_dc_true += p_dc_true
            
            # --- 2. Calcul Dowell pur (Carré) ---
            delta_i = layer.delta(freq)
            g1, g2 = self._dowell_G(delta_i)
            H_i = self.field_at_boundary(i, currents_rms)
            alpha_i = self.alpha(i, currents_rms)
            
            bracket = (1.0 + alpha_i**2) * g1 - 4.0 * alpha_i * g2
            left_part = layer.MLT / (layer.height * layer.porosity * layer.sigma)
            p_ac_dowell = left_part * (H_i**2) * bracket * layer.bw
            
            # --- 3. Extraction du vrai facteur Fr ---
            r_dc_dowell = layer.dc_resistance_dowell()
            p_dc_dowell = r_dc_dowell * (layer_current**2)
            
            Fr = p_ac_dowell / p_dc_dowell if p_dc_dowell > 0 else 1.0
            
            # --- 4. Application du Fr à la géométrie ronde ---
            p_ac_true = Fr * p_dc_true
            total_p_ac_true += p_ac_true
            
        return total_p_dc_true, total_p_ac_true

if __name__ == '__main__':
    freq_sw = 100e3                     
    bw_total = pi * 7.62e-3             
    mlt = 17.5e-3                       
    
    currents = {
        WINDING_TYPE.PRIMARY: 0.3,
        WINDING_TYPE.SECONDARY: 0.5,
    }

    flyback_transfo = Dowell_Winding_Structure()

    d_pri = 0.3e-3
    n_pri = 67
    porosity_pri = porosity(n_pri, d_pri, bw_total)
    
    flyback_transfo.add_layer(Layer(
        name="Primaire",
        number_of_turns=n_pri,
        bw=bw_total,
        height=d_pri,
        MLT=mlt,
        winding_type=WINDING_TYPE.PRIMARY,
        polarity=POLARITY.POSITIVE,
        porosity=porosity_pri
    ))

    bw_sec = bw_total 
    bw_aux = bw_total 
    
    d_sec_aux = 0.4e-3
    
    porosity_sec = porosity(26, d_sec_aux, bw_sec)

    flyback_transfo.add_layer(Layer(
        name="Secondaire",
        number_of_turns=26,
        bw=bw_sec,
        height=d_sec_aux,
        MLT=mlt,
        winding_type=WINDING_TYPE.SECONDARY,
        polarity=POLARITY.NEGATIVE,
        porosity=porosity_sec
    ))


    p_dc = flyback_transfo.dc_loss(currents)
    p_ac_100k = flyback_transfo.loss_at_frequency(freq_sw, currents)
    fr = p_ac_100k / p_dc if p_dc > 0 else 1.0
    delta_cu = skin_depth(freq_sw) * 1e6 
    
    print("=== RÉSULTATS GLOBAUX ===")
    print(f"Épaisseur de peau à 100 kHz : {delta_cu:.1f} µm")
    print(f"Pertes DC totales            : {p_dc*1000:.2f} mW")
    print(f"Pertes AC (à 100 kHz)        : {p_ac_100k*1000:.2f} mW")
    print(f"Facteur Dowell Global        : {fr:.3f}\n")
    
    print("=== RÉSISTANCES PAR ENROULEMENT ===")
    for i, layer in enumerate(flyback_transfo.list_of_layers):
        layer_current = currents.get(layer.winding_type, 0.0)
        rdc = layer.dc_resistance()
        
        delta_i = layer.delta(freq_sw)
        g1, g2 = flyback_transfo._dowell_G(delta_i)
        H_i = flyback_transfo.field_at_boundary(i, currents)
        alpha_i = flyback_transfo.alpha(i, currents)
        bracket = (1.0 + alpha_i**2) * g1 - 4.0 * alpha_i * g2
        left_part = layer.MLT / (layer.height * layer.porosity * layer.sigma)
        p_ac_layer = left_part * (H_i**2) * bracket * layer.bw
        
        rac = p_ac_layer / (layer_current**2) if layer_current > 0 else 0.0
        
        print(f"- {layer.name} :")
        print(f"    R_dc = {rdc*1000:.2f} mΩ")
        print(f"    R_ac = {rac*1000:.2f} mΩ (@ 100kHz)")
        print(f"    Fr   = {rac/rdc if rdc > 0 else 1:.3f}")