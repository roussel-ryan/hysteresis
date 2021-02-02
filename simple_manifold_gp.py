import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import copy


def f_int(x, A, B):
    return torch.sin(A * x ** 2 + B)
    

class SimpleManifoldGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleManifoldGPModel, self).__init__(train_x, train_y, likelihood)

        #register parameters
        self.register_parameter('A', torch.nn.parameter.Parameter(1.0*torch.ones(1)))
        self.register_parameter('B', torch.nn.parameter.Parameter(1.0*torch.ones(1)))

        #lengthscale constraint
        lconstr = gpytorch.constraints.Interval(0.0,10.0)
        
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint = lconstr))

        #freeze lengthscale parameter
        #self.covar_module.base_kernel.raw_lengthscale.requires_grad = False

        
    def forward(self, x):


        #transform x to intermediate space
        inter_x = f_int(x, self.A, self.B)
        
        mean = self.mean_module(inter_x)
        covar = self.covar_module(inter_x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    
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

        if i % 100 == 0:
            print('Iter %d - Loss: %.3f - Best loss %.3f' % (i + 1, loss.item(), best_loss))
            
        optimizer.step()

    #set model params to the best
    model.load_state_dict(best_param)
    

if __name__ == '__main__':

    #testing suite
    true_A = 5.0 * np.pi
    true_B = 0.5
    x = torch.linspace(0,1,20)
    x_int = f_int(x, true_A, true_B)
    y = x_int**3 + torch.rand(x_int.shape)*0.1**2
    y = y.flatten()

    #L = torch.Tensor([1.,5.]).reshape((2,1))
    
    lk = gpytorch.likelihoods.GaussianLikelihood()

    model = SimpleManifoldGPModel(x, y, lk)

    train_model(model,lk, x, y, iter_steps = 550, lr = 0.1)
    #print(model.state_dict())
    print(model.A.data)
    print(model.B.data)
    print(model.covar_module.base_kernel.lengthscale)
    #print(model.B.data)

    fit_A = model.A.data.detach()
    fit_B = model.B.data.detach()
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().squeeze()
    
    n = 200
    x_test = torch.linspace(0,1,n)
    x_int_test = f_int(x_test, true_A, true_B)
    #xx,yy = torch.meshgrid(x,x)
    #pts = torch.stack((xx.flatten(), yy.flatten())).transpose(0,1)
    
    model.eval()

    pred = lk(model(x_test))

    mean = pred.mean

    lower, upper = pred.confidence_region()
    
    fig,axes = plt.subplots(3,1)
    ax = axes[0]
    ax.plot(x_test, mean.detach(), label = 'Posterior mean')
    ax.plot(x,y,'+',label = 'Observations')
    ax.fill_between(x_test, lower.detach(), upper.detach(), lw = 0, alpha = 0.5, fc = 'C0', label = 'Posterior variance')

    ax.set_xlabel('h')
    ax.set_ylabel('y')
    ax.legend()
    
    #fig,ax = plt.subplots()
    axes[1].plot(x_int_test,mean.detach(),'+')
    axes[1].set_xlabel('x = g(h)')
    axes[1].set_ylabel('Posterior mean')
    
    #fig,ax = plt.subplots()
    axes[2].plot(x_test, f_int(x_test, true_A, true_B),label = 'Ground truth')
    axes[2].plot(x_test, f_int(x_test, fit_A, fit_B),ls='--', label = 'MLE fit')

    axes[2].set_xlabel('h')
    axes[2].set_ylabel('g(h)')
    axes[2].legend()
    
    plt.show()
    
