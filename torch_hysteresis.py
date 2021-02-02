import torch
import numpy as np

import matplotlib.pyplot as plt

class PreisachModel:
    def __init__(self, H, **kwargs):
        #specify grid size
        self.grid_size = kwargs.get('grid_size',100)
        
        #positive and negative saturation levels
        self.alpha_0 = kwargs.get('alpha_0',1.0)
        self.beta_0 = kwargs.get('beta_0',-1.0)
        assert self.alpha_0 > self.beta_0

        #create grid
        b = np.linspace(self.beta_0, -self.beta_0, self.grid_size)
        a = np.linspace(-self.alpha_0, self.alpha_0, self.grid_size)
        bb,aa = np.meshgrid(b, a)

        #create grid tensors
        self.beta = torch.from_numpy(bb)
        self.alpha = torch.from_numpy(aa)

        #create grid for mu (hysterion density)
        #self.mu = torch.zeros((self.grid_size, self.grid_size))

        #create grid for the state history
        #size: (tsteps+1, grid_size, grid_size)
        self.initial_state = -1 * torch.tril(torch.ones((1, self.grid_size, self.grid_size)))
        self.state = self.initial_state
        
        #track the applied field values H
        self.H = H

    def get_grid(self):
        #returns copy of grid points in alpha/beta space
        t = torch.vstack((torch.flatten(self.beta),torch.flatten(self.alpha)))
        return torch.transpose(t,0,1)
        

    def calculate_field_response(self, mu):
        assert mu.shape == self.state[0].shape

        #given the states, calculate the field response as a function of mu
        n_steps = self.state.shape[0]

        #results container
        results = torch.empty((n_steps))
        for i in range(n_steps):
            results[i] = torch.sum(mu * self.state[i])

            
        return results[1:]
        
    def propagate_states(self):
        #recalculate the state history given current applied field history H
        n_steps = self.H.shape[0]
        
        for i in range(n_steps):
            if i == 0:
                dH = 1
            else:
                dH = self.H[i] - self.H[i - 1]

            new_state = self.update_state(self.H[i], dH, self.state[i])
            self.state = torch.cat((self.state,new_state),0)

    

    def update_state(self, H, dH, state):
        if dH > 0:
            #determine locations where we want to set the state to +1
            flip = torch.where(H > self.alpha,
                               torch.ones_like(self.alpha),
                               torch.zeros_like(self.alpha))
            #print(torch.nonzero(flip))
            new_state = state.clone()

            new_state[torch.nonzero(flip, as_tuple=True)] = 1
            new_state = new_state.reshape(1,self.grid_size,self.grid_size)
            
        elif dH < 0:
            #determine locations where we want to set the state to -1
            flip = torch.where(H < self.beta,
                               torch.ones_like(self.beta),
                               torch.zeros_like(self.beta))
            #print(torch.nonzero(flip))
            new_state = state.clone()

            new_state[torch.nonzero(flip, as_tuple=True)] = -1
            new_state = new_state.reshape(1,self.grid_size,self.grid_size)

            
        else:
            new_state = state.clone()
            new_state = new_state.reshape(1,self.grid_size,self.grid_size)

            
        #get upper triangle
        return torch.tril(new_state)

    def visualize(self):
        for i in range(self.state.shape[0]):
            plt.figure()
            plt.imshow(self.state[i].numpy(),vmin=-1,vmax=1,origin='lower')

#testing
if __name__ == '__main__':
    H = torch.tensor([-2.75, -0.25, 0.0, -0.5, 2.5])
    m = PreisachModel(H)

    mu = torch.ones((m.grid_size, m.grid_size))
    mu = mu / torch.sum(mu)
    
    m.propagate_states()
    m.visualize()

    print(m.calculate_field_response(mu))
    
    plt.show()
