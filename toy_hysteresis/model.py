import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

from hysteresis_gp import densities
from hysteresis_gp import preisach
import generator

import gpytorch

class HysteresisExact(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, lk, pmodel):
        super(HysteresisExact, self).__init__(train_x, train_y, lk)

        #register parameters
        self.register_parameter('mu', torch.nn.parameter.Parameter(0.1 * torch.ones(2)))
        self.register_parameter('cov_diag',torch.nn.parameter.Parameter(1.5 * torch.ones(2)))

        #lengthscale constraint
        #lconstr = gpytorch.constraints.Interval(0.0,1.0)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.pmodel = pmodel
        
        self.first_run = True
        
    def forward(self, x):        
        self.pmodel.set_applied_field(x)
        self.pmodel.propagate_states()

        dens = densities.MultiVariateNormal(self.mu, torch.diag(self.cov_diag))
        B = self.pmodel.calculate_magnetization(dens).reshape(-1,1)
    
        mean = self.mean_module(B)
        covar = self.covar_module(B)

        return gpytorch.distributions.MultivariateNormal(mean, covar)
            
        
def quad_focusing(K):
    return (K - 0.25)**2

def main():
    t, current, pmodel = generator.generate_magnetization_model(100)

    #create hysterion density object
    mu = torch.zeros(2) + 0.5
    cov = torch.tensor(((1.0,0.0),(0.0,0.75)))
    hdens = densities.MultiVariateNormal(mu, cov)

    #generate magnetization response
    response = pmodel.calculate_magnetization(hdens)

    #plot magnet response
    fig,ax = plt.subplots()
    #ax.plot(current, response)
    ax.plot(t, response)

    #plot beam response
    fig2, ax2 = plt.subplots(2,1)
    sigma = quad_focusing(response)
    ax2[0].plot(current, sigma, 'C1+')
    ax2[1].plot(response, sigma, '+')


    #create and fit model
    train_x = torch.from_numpy(current).reshape(-1,1)
    train_y = sigma
    
    lk = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = HysteresisExact(train_x, train_y, lk, pmodel)

    
    train_model(gp_model, lk, train_x, train_y, iter_steps = 300,lr = 0.4)

    
    #evaluate model
    gp_model.eval()

    #test_x = torch.linspace(0.,1.,50).reshape(-1,1)

    print(gp_model.state_dict())
    print(gp_model(train_x))
    with torch.no_grad():
        pred = lk(gp_model(train_x))

    mean = pred.mean.detach()
    lower, upper = pred.confidence_region()

    ax2[0].plot(train_x, mean)
    ax2[0].fill_between(train_x.flatten(), lower.detach(), upper.detach(), lw = 0, alpha = 0.25)
    
def train_model(model, likelihood, x, y, iter_steps = 250, lr = 0.1):    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = 10000
    for i in range(iter_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        if loss.item() < best_loss:
            best_param = copy.deepcopy(model.state_dict())
            best_loss = loss.item()

        if i % 10 == 0:
            print('Iter %d - Loss: %.3f - Best loss %.3f' % (i + 1, loss.item(), best_loss))
            
        optimizer.step()

    #set model params to the best
    model.load_state_dict(best_param)

    
main()
plt.show()
