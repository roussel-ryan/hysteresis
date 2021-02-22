import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from hysteresis import densities
from hysteresis import preisach
from hysteresis import models
import generator

import gpytorch

        
def quad_focusing(K):
    return (K - 0.25)**2

def main():
    n_pts = 100
    t, current, p_model = generator.generate_magnetization_model(n_pts)

    #create ground truth hysterion density object
    mu = torch.tensor((0.0,0.0))
    cov = torch.tensor(((0.5,0.0),(0.0,1.0)))
    hdens = densities.MultiVariateNormal(mu, cov)

    #create prior hysterion density object
    mu = torch.tensor((0.0, 0.0))
    cov = torch.tensor((0.75, 0.75))
    prior_dens = densities.MultiVariateNormal(mu, diag_cov = cov)

    
    #generate magnetization + beam response
    response = p_model.calculate_magnetization(hdens)
    sigma = quad_focusing(response) + torch.randn(response.shape) * 0.1**2

    #plot sigma(t)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,sigma.detach())
    ax[1].plot(current, response.detach())
    
    
    #create training/testing sets
    n_train = int(1.0 * n_pts)
    train_x = torch.from_numpy(current[:n_train]).reshape(-1,1)
    train_y = sigma[:n_train]
    
    test_x = torch.from_numpy(current).reshape(-1,1)[-10:]
    test_y = sigma[-10:]
    

    
    lk = gpytorch.likelihoods.GaussianLikelihood()
    h_module = models.Hysteresis(p_model, prior_dens)
    gp_model = models.HysteresisExact(train_x, train_y, lk, h_module)
    
    #train and/or load model
    print('start model training')

    #set cov to not trainable
    gp_model.hyst_module.density_model.mu.requires_grad = False
    #gp_model.hyst_module.density_model.diag_cov.requires_grad = False
    models.train_model(gp_model, lk, train_x, train_y, iter_steps = 150,lr = 0.1)
    gp_model.load_state_dict(torch.load('model.pth'))
    
    #evaluate model
    gp_model.eval()
    lk.eval()
    
    #test_x = torch.linspace(0.,1.,50).reshape(-1,1)

    print(gp_model.state_dict())
    #with torch.no_grad():
    #    pred = lk(gp_model(test_x))

    #mean = pred.mean.detach()
    #lower, upper = pred.confidence_region()

    #plot beam response
    #fig2, ax2 = plt.subplots(2,1)
    #ax2[0].plot(test_x.detach(), test_y.detach(), 'C1+')
    #ax2[1].plot(response.detach(), sigma.detach(), '+')

    
    
    #ax2[0].plot(test_x.detach(), mean)
    #ax2[0].fill_between(test_x.detach().flatten(), lower.detach(), upper.detach(), lw = 0, alpha = 0.25)

    #fig3,ax3 = plt.subplots()
    #ax3.plot(current[-10:], response.detach()[-10:])
    
    #use the trained preiscah model to predict the next magnetization step
    h_new = torch.linspace(-1.0,1.0,60).reshape(-1,1).double()
    response_new = p_model.predict_magnetization(h_new, hdens)
    ground_new = quad_focusing(response_new) + torch.randn(response_new.shape) * 0.1 ** 2

    if 1:
        with torch.no_grad():
            next_pred = lk(gp_model(h_new))
            #B = gp_model.predict(h_new)
        
            m = next_pred.mean
            l, u = next_pred.confidence_region()

            fig4, ax4 = plt.subplots(2,1)
            ax4[0].plot(h_new, m)
            ax4[0].fill_between(h_new.flatten(), l.flatten() ,u.flatten() ,alpha=0.25, lw=0)
            ax4[0].plot(h_new,ground_new,'+')
            ax4[1].plot(h_new, u - l)
            
main()
plt.show()
