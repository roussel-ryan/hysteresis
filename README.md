# hysteresis modeling

In this project we attempt to simultaneously model the joint behavior of hysteresis in focusing magnets as well as the beam response to those hysteresis effects using pytorch and gpytorch

The joint model is defined as follows:
- input variable: applied magnetic field (H) / applied current (I) (note: H is proportional to I)
- a pytorch module that maps the input to an intermediate variable, representing the resulting magnetic field (B)
-- the pytorch module as several trainable parameters, which describes the hysterion density in the Preiscah model
- a ExactGP gpytorch module which maps the intermediate variable to the observable
-- the ExactGP represents a gaussian process with a Matern kernel which has standard trainable hyperparameters
- a Gaussian likelihood

We train the model parameters via maximizing the marginal log likelihood using the ADAM algorithm.

This process mirrors manifold GPs (https://arxiv.org/abs/1402.5876).

File structure
- hysteresis -- modeling package
-- hysteresis/models.py -- contains ExactGP models that themselves contain hysteresis modules to create the joint model
-- hysteresis/preisach.py -- contains pytorch implementation of the Preisach model
-- hysteresis/densities.py -- contatins module that calculates the hysterion densities as well as trainable parameters
-- hysteresis/simple_manifold_gp.py -- contains simple implementation of manifold GPs

- toy_hysteresis -- simple hysteresis example
-- toy_hysteresis/fit_model.py -- main example file that trys to fit a hysteresis/GP joint model using a simulated magnet and a simple beam response function (f(x) = (x - 0.25)**2, where x is the intermediate variable)
-- toy_hysteresis/generator.py -- generates data for fit_model.py
