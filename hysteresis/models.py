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
        '''
        NOTE: remember that the argument passed to x here is a union between training and test data
              when self.training = False
        - we need to re-do the hysteresis model (call calculate_magnetization) for the training data
          but when we predict future values we need to call predict_magnetization

        '''        
        if self.training:
            B = self.hyst_module(x)
        else:
            #note: to allow batch training self.train_inputs is a tuple of tensors, need first index
            # not the case for passing x!
            n_train = self.train_inputs[0].shape[0]
            #print(f'x:{x}')

            B_train = self.hyst_module(x[:n_train])
            #print(f'B_train:{B_train}')
            
            B_test = self.hyst_module.predict(x[n_train:]).float()
            #print(f'B_test:{B_test}')
            
            #concat
            B = torch.cat((B_train, B_test),axis = 0)
            #print(f'B:{B}')
            
        mean = self.mean_module(B)
        covar = self.covar_module(B)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def get_manifold_model(self):
        #from this model create a copy GP that predicts from manifold space, not input space

        #get manifold inputs and outputs
        #note will not work for batch mode!
        manifold_inputs = self.hyst_module(self.train_inputs[0])
        manifold_outputs = self.train_targets

        #define new model - use deep copies so modifications to new model do not change orig model
        manifold_lk = copy.deepcopy(self.likelihood)
        manifold_model = ExactGP(manifold_inputs, manifold_outputs, manifold_lk)

        #set model cov and mean function to copy
        manifold_model.mean_module = copy.deepcopy(self.mean_module)
        manifold_model.covar_module = copy.deepcopy(self.covar_module)

        return manifold_model
        
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
    
