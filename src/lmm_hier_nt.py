import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.distributions import constraints
from pyro.distributions.util import eye_like

import numpy as np
import pandas as pd
from dataclasses import dataclass
import seabass

def convertr(hyperparam, name, device): 
    return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = device) 

def model_base(
    data,
    slope_noise = 1., 
    log_sigma_noise_mean = 0.,
    log_sigma_noise_std = 1.,
): 
    """ linear mixed model for NT guides. No true effect. 
    
    log2FC = (guide_random_slope) [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) for each observation
    guide_random_slope ~ Normal(0, slope_noise^2) for each (guide,replicate) pair
    
    Parameters
    ----------
    Data: a seabass.Data object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """
    
    log_sigma_noise_mean = convertr(log_sigma_noise_mean, "log_sigma_noise_mean", device = data.device)
    log_sigma_noise_std = convertr(log_sigma_noise_std, "log_sigma_noise_std", device = data.device)
    
    log_sigma_noise = pyro.sample(
        "log_sigma_noise",
         dist.Normal(log_sigma_noise_mean, log_sigma_noise_std).expand([data.num_guides]).to_event(1)
    )

    slope_noise = convertr(slope_noise, "slope_noise", device = data.device)
    
    mean = torch.zeros_like(data.logFC)
    
    if data.multiday: 
        random_slope = pyro.sample("random_slope", # ideally would probably integrate over this
            dist.Normal(0., slope_noise).expand([data.num_guides,data.num_replicates]).to_event(2)
        )
        mean = (random_slope[data.guide_indices, data.replicate]) * data.timepoint 
    
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, log_sigma_noise.exp()[data.guide_indices]), obs = data.logFC)


def guide_structured(data):
    
    assert(data.multiday) # only makes sense in this setting
    
    device = data.device
        
    def conc_helper(name, init = 1.,  constraint = constraints.positive):    
        param = pyro.param(name + "_param", lambda: torch.tensor(init, device = device), constraint=constraint)
        return pyro.sample(name, dist.Delta(param))
    
    log_sigma_noise_mean = conc_helper("log_sigma_noise_mean", init = 0., constraint = constraints.real)
    log_sigma_noise_std = conc_helper("log_sigma_noise_std")
    slope_noise = conc_helper("slope_noise")
    
    log_sigma_noise_loc = pyro.param('log_sigma_noise_loc', lambda: torch.zeros(data.num_guides, device = device))
    log_sigma_noise_scale = pyro.param(
        'log_sigma_noise_scale', 
        lambda: torch.ones(data.num_guides, device = device), 
        constraint=constraints.positive)
    log_sigma_noise = pyro.sample(
        "log_sigma_noise", 
        dist.Normal(log_sigma_noise_loc, log_sigma_noise_scale).to_event(1))
    
    P = data.num_replicates
    loc = pyro.param("loc",torch.zeros(data.num_guides,P))
    scale = pyro.param(
        "scale",
        torch.full_like(loc, 0.1), 
        constraint = constraints.positive
    )
    scale_tril = pyro.param(
        "scale_tril",
        eye_like(loc, P).repeat([data.num_guides,1,1]), 
        constraint = constraints.unit_lower_cholesky
    )
    random_slope = pyro.sample( # will be guides x P
        "random_slope", 
        dist.MultivariateNormal(loc, scale_tril = scale[...,None] * scale_tril).to_event(1), 
    )

def extract_params(to_optimize):
    
    to_return = { 
        "sigma_noise_mean" : pyro.param("log_sigma_noise_mean_param").exp(),
        "log_sigma_noise_std" : pyro.param("log_sigma_noise_std_param"),
        "slope_noise_std" : pyro.param("slope_noise_param"),
        "sigma_noise_std" : pyro.param('log_sigma_noise_loc').exp(),
        "log_sigma_noise_se" : pyro.param('log_sigma_noise_scale'),
        "random_slope" : pyro.param("loc"),
        "random_slope_se" : pyro.param("scale") * pyro.param("scale_tril").diagonal(dim1=-1,dim2=-2)
    }
    
    return { k:v.detach().cpu().numpy() for k,v in to_return.items() }
        
def fit(
    data,
    iterations = 1000,
    print_every = 100,
    lr = 0.03,
    slope_noise = None, # set to None to learn, or fix to a value
    log_sigma_noise_mean = None,
    log_sigma_noise_std = None,
): 
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    model = lambda data:  model_base(
        data, 
        log_sigma_noise_mean = dist.Cauchy(one*0.,one) if (log_sigma_noise_mean is None) else log_sigma_noise_mean, 
        log_sigma_noise_std = dist.HalfCauchy(one) if (log_sigma_noise_std is None) else log_sigma_noise_std, 
        slope_noise = dist.HalfCauchy(one) if (slope_noise is None) else slope_noise
    )
    
    to_optimize = [
        "log_sigma_noise_mean",
        "log_sigma_noise_std",
        "slope_noise"
    ]
    
    #guide = AutoGuideList(model)
    #guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
    #guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))
    
    #guide = guide_mean_field
    
    guide = lambda data: guide_structured(data)
    
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    # train/fit model
    pyro.clear_param_store()
    losses = []
    for j in range(iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(data)
        losses.append(loss)
        if j % print_every == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data.guide_indices)))
            
    posterior = extract_params(to_optimize)
    
    return( model, guide, losses, posterior )

