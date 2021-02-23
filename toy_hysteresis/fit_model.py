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
    n_pts = 20
    t, current, p_model = generator.generate_magnetization_model(n_pts)

    #create ground truth hysterion density object
    mu = torch.tensor((-0.0,0.0))
    cov = torch.tensor(((0.25,0.0),(0.0,1.5)))
    hdens = densities.MultiVariateNormal(mu, cov)

    #create prior hysterion density object
    mu = torch.tensor((0.0, 0.0))
    cov = torch.tensor((0.75, 0.75))
    prior_dens = densities.MultiVariateNormal(mu, diag_cov = cov)
    
    #generate magnetization + beam response
    response = p_model.calculate_magnetization(hdens)
    sigma = quad_focusing(response) + torch.randn(response.shape) * 0.1**2

    print(response)

    
    no_hyst = quad_focusing(current)
    
    #plot sigma(t)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t, sigma.detach(), label = 'Hysteresis on')
    ax[0].plot(t, no_hyst, label = 'Hysteresis off')
    ax[0].legend()
    ax[0].set_ylabel('Beam size')
    ax[0].set_xlabel('time step')

    #ax[1].plot(current, sigma.detach() - no_hyst, '+')
    #ax[1].set_ylabel('Beam size error')
    #ax[1].set_xlabel('H')

    
    ax[1].plot(current, response.detach())
    ax[1].set_xlabel('H')
    ax[1].set_ylabel('B')
    
    
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
    #models.train_model(gp_model, lk, train_x, train_y, iter_steps = 150,lr = 0.05)
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

    fig5, ax5 = plt.subplots(1,3)

    
    #use the trained preiscah model to predict the next magnetization step
    h_new = torch.linspace(-1.0,1.0,30).reshape(-1,1).double()
    response_new = p_model.predict_magnetization(h_new, hdens)

    plt.figure()
    ax5[0].plot(h_new, response_new.detach(), label = '$B(H_{t+1})$ - Ground truth')
    ax5[0].plot(h_new, gp_model.hyst_module.predict(h_new).detach(), label = '$B(H_{t+1})$ - Prediction')

    ax5[0].plot(current, response.detach(),'-+', c = 'C0', alpha = 0.25,label = '$B(H_{0:t})$')


    ax5[0].set_ylabel('$B$')
    ax5[0].set_xlabel('$H$')
    ax5[0].legend()
    
    ground_new = quad_focusing(response_new)# + torch.randn(response_new.shape) * 0.1 ** 2

    no_hysteresis = quad_focusing(h_new)

    #get manifold model
    manifold_model = gp_model.get_manifold_model()
    manifold_model.eval()
    manifold_model.likelihood.eval()

    print(gp_model.prediction_strategy)
    
    if 1:
        with torch.no_grad():
            mani_pred = manifold_model.likelihood(manifold_model(h_new.float()))

            m = mani_pred.mean
            l, u = mani_pred.confidence_region()

            ax5[1].plot(manifold_model.train_inputs[0], manifold_model.train_targets,'+', label = 'Samples', zorder = 10)
            ax5[1].plot(h_new, m, label = 'Posterior mean')
            ax5[1].fill_between(h_new.flatten(), l.flatten() ,u.flatten() ,alpha=0.25, lw=0, fc = 'C1')
            ax5[1].set_ylabel('Beam size')
            ax5[1].set_xlabel('B')
            ax5[1].legend()
    
    if 1:
        with torch.no_grad():
            next_pred = gp_model.likelihood(gp_model(h_new))
            #B = gp_model.predict(h_new)
        
            m = next_pred.mean
            l, u = next_pred.confidence_region()

            #fig4, ax4 = plt.subplots()
            ax5[2].plot(h_new, m, c = 'C1', label = 'Posterior mean')
            ax5[2].fill_between(h_new.flatten(), l.flatten() ,u.flatten() ,alpha=0.25, fc = 'C1', lw=0)
            ax5[2].plot(h_new, ground_new, c = 'C0', label = 'Ground truth')
            ax5[2].plot(h_new, no_hysteresis,'--', c = 'C2', label = 'Hysteresis off')
            ax5[2].set_xlabel('$H_{t+1}$')
            ax5[2].set_ylabel('Beam size')
            ax5[2].legend()

            
            #ax4.plot(h_new, u - l)
            
main()
plt.show()
