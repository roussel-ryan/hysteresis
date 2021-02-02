import matplotlib.pyplot as plt
import numpy

import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


import torch_hysteresis

def base_function(x):
    return x ** 2


def main():
    H = torch.rand(5, 1)
    B = 

    train_Y = base_function(train_X)

    train_Y = standardize(train_Y)

    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)


main()
