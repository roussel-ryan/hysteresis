import numpy as np
import torch

from hysteresis_gp import preisach

def generate_current_data(n_pts):
    #generate the sequence of current settings (in arb units)
    t = np.linspace(0,1,n_pts)
    current = -1.0*(1 - 0.25*t)*np.sin(2 * np.pi * t / 0.25 + np.pi/2)
    return t, current

def generate_magnetization_model(n_pts = 20):
    t, current = generate_current_data(n_pts) 
    pmodel = preisach.PreisachModel(torch.from_numpy(current))

    pmodel.propagate_states()
    return t, current, pmodel
    
