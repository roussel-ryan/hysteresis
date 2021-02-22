import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

from . import densities
from . import preisach

import gpytorch

#hybrid hysteresis/GP
class HybridGP(torch.nn.Module):
    def __init__(self,
                 hysteresis_model,
                 density_model,
                 train_x, train_y, lk,
                 priors = None):

        super(HybridGP, self).__init__()

        self.h_module = Hysteresis(hysteresis_model, density_model)
        self.gp_module = ExactGP(train_x, train_y, lk)

    def forward(x):
        B = self.h_module(x)
        return self.gp_module(B)

        
#impliment a hysteresis module as a subclass of torch.nn
class Hysteresis(torch.nn.Module):
    def __init__(self,
                 hysteresis_model,
                 density_model,
                 priors = None):
        
        super(Hysteresis, self).__init__()

        #register parameters from density model
        #for name, item in density_model.params.items():
        #    self.register_parameter(name,item)
        
        self.density_model = density_model
        self.hysteresis_model = hysteresis_model
        
    def forward(self, x):
        self.hysteresis_model.set_applied_field(x)
        self.hysteresis_model.propagate_states()

        return self.hysteresis_model.calculate_magnetization(
            self.density_model).reshape(-1,1)

    def predict(self, x):
        return self.hysteresis_model.predict_magnetization(x,
                                                           self.density_model).reshape(-1,1)

class HysteresisExact(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, lk, hyst_module):
        super(HysteresisExact, self).__init__(train_x, train_y, lk)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.hyst_module = hyst_module
        
        
    def forward(self, x):        
        if self.training:
            B = self.hyst_module(x)
        else:
            B = self.hyst_module.predict(x).float()
        
        mean = self.mean_module(B)
        covar = self.covar_module(B)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def predict(self, x):
        B = self.hyst_module.predict(x)
        print(B)
        mean = self.mean_module(B)
        covar = self.covar_module(B)

        return gpytorch.distributions.MultivariateNormal(mean, covar)
        
        
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, lk):
        super(ExactGP, self).__init__(train_x, train_y, lk)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):                
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
        

def train_model(model, likelihood, x, y, iter_steps = 250, lr = 0.1, fname = ''):    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    
    # Use the adam optimizer
    for name, param in model.named_parameters():
        print(f'{name}:{param.requires_grad}')

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = 10000
    for i in range(iter_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward(retain_graph=True)
        if loss.item() < best_loss:
            best_param = copy.deepcopy(model.state_dict())
            best_loss = loss.item()

        if i % 10 == 0:
            print('Iter %d - Loss: %.3f - Best loss %.3f' % (i + 1, loss.item(), best_loss))
            
        optimizer.step()

    #set model params to the best
    model.load_state_dict(best_param)

    torch.save(model.state_dict(), f'{fname}model.pth')
    
