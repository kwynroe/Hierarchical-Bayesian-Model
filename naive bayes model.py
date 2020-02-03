# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:00:28 2020

@author: KEITH618
"""
import torch
import pyro
import pyro.distributions as dist
import pyro.contrib

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


pyro.enable_validation(True)
pyro.clear_param_store()

### Specify constraints & observed data ###
bag_samples = 500
colors = 2
data = torch.tensor([
                     [500.,0.],[500.,0.],[500.,0.],[500.,0.],[500.,0.],
                     [0.,500.],[0.,500.],[0.,500.],[0.,500.],[0.,500.],
                     [500.,0.],[500.,0.],[500.,0.],[500.,0.],[500.,0.],
                     [0.,500.],[0.,500.],[0.,500.],[0.,500.],[0.,500.],
                     [500.,0.],[500.,0.],[500.,0.],[500.,0.],[500.,0.],
                     [0.,500.],[0.,500.],[0.,500.],[0.,500.],[0.,500.],
                     [500.,0.],[500.,0.],[500.,0.],[500.,0.],[500.,0.],
                     [0.,500.],[0.,500.],[0.,500.],[0.,500.],[0.,500.],
                     ])

def naive_model(data = data):
       with pyro.plate("thetas", len(data)):
               theta = pyro.sample("bag_theta", dist.Dirichlet(torch.ones(colors)))
               pyro.sample("draws", dist.Multinomial(bag_samples, theta), obs = torch.tensor(data))

def naive_guide(data = data):
        conc = pyro.param("concentrations", torch.ones(colors))
        with pyro.plate("thetas", len(data)):
             pyro.sample("bag_theta", dist.Dirichlet(conc))

def train(model, guide):
        pyro.clear_param_store()
        adam_params = {"lr": 0.0005}
        optimizer = optimizer = Adam(adam_params)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        #alphas, betas, losses = [], [],[]
        n_steps = 10000
        for step in range(n_steps):
                #print(step)
                svi.step(data)
                if n_steps % 100 == 0:
                        print(step)
                        
train(naive_model, naive_guide)
new_bag = pyro.sample("new_bag", dist.Dirichet(pyro.param("concentrations")))
