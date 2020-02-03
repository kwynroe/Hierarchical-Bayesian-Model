# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:44:13 2020

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

### Learning Over-hypotheses , i.e posterior over out hyper-parameters ###
def model(data = data):
    alpha = pyro.sample("alpha", dist.Gamma(1.0,1.0))
    betas = pyro.sample("betas", dist.Dirichlet(torch.ones(colors)))
    with pyro.plate("thetas", len(data)):
         pyro.sample("test", \
                     dist.DirichletMultinomial((alpha*betas), bag_samples),
                     obs = torch.tensor(data))
  
def guide(data = data):
        a = pyro.param("a", torch.tensor(1.0))
        b = pyro.param("b", torch.tensor(1.0))
        conc = pyro.param("q_conc", torch.ones(colors))
        pyro.sample("alpha", dist.Gamma(a,b))
        pyro.sample("betas", dist.Dirichlet(conc))

### Define inference loop ###
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
                       
train(model,guide)

### Save parameters ###
k = pyro.param("a")
rate = pyro.param("b")
pop_dist = pyro.param("q_conc")

mean_alpha = dist.Gamma(k,rate).mean
dir_params = mean_alpha*pop_dist
dir_params = dir_params.tolist()
### Given posterior of hyper-parameters, sample new bag and infer posterior

draws = [0.,1.]

def lower_model(draws = draws):
       theta = pyro.sample("bag_theta", dist.Dirichlet(torch.tensor(dir_params)))
       return pyro.sample("draws", dist.Multinomial(sum(data), theta), obs = torch.tensor(data))

def lower_guide(data = data):
        conc = pyro.param("conc", torch.ones(colors))
        return pyro.sample("bag_theta", dist.Dirichlet(conc))
 
train(lower_model,_guide)
new_bag = dist.Dirichlet(pyro.param("conc")).mean

### Finally - predict! ###
guess = pyro.sample("guess", dist.Categorical(new_bag))
'''
def naive_model(data = data):
       theta = pyro.sample("bag_theta", dist.Dirichlet(torch.ones(colors)))
       return pyro.sample("draws", dist.Multinomial(sum(data), theta), obs = torch.tensor(data))

def naive_guide(data = data):
        conc = pyro.param("control_conc", torch.ones(colors))
        return pyro.sample("bag_theta", dist.Dirichlet(conc))
'''
train(control_model, control_guide)
