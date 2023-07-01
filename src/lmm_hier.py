import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, init_to_value
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.distributions import constraints
from pyro.distributions.util import eye_like

from scipy.stats import norm
import scipy.stats

import statsmodels.stats.multitest 

import numpy as np
import pandas as pd
from dataclasses import dataclass
import seabass

def robustifier(original_func, max_attempts, *args, **kwargs):
    
    attempts = 0
    while attempts < max_attempts:
        try:
            result = original_func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {str(e)}")
            attempts += 1

    raise Exception("Function failed after maximum attempts")

def convertr(hyperparam, name, device): 
    return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = device) 

def get_dist(name, t_df, device): 
    if type(t_df) == float and np.isinf(t_df):
        return dist.Normal
    elif type(t_df) == float and t_df == 0.: # denotes Laplace
        return dist.Laplace
    elif type(t_df) == float and t_df == 1.: # denotes Cauchy
        return dist.Cauchy
    else:
        t_df = convertr(t_df, "%s_t_df" % name, device = device)
        return lambda m,s: dist.StudentT(t_df, m, s)

def model_base(
    data,
    NT_model, 
    hierarchical_noise, 
    hierarchical_slope,
    per_gene_variance,
    guide_std = 1., 
    log_guide_std_mean = 0.,
    log_guide_std_std = 1.,
    slope_noise = 1., 
    log_slope_noise_mean = 0.,
    log_slope_noise_std = 1.,
    sigma_noise = 1., 
    log_sigma_noise_mean = 0.,
    log_sigma_noise_std = 1.,
    t_df = 3.,
    noise_t_df = np.inf,
    slope_t_df = np.inf
): 
    """ linear mixed model for guides. 
    
    guide_score ~ Normal(0, guide_std^2) for each guide
    log2FC = (guide_score + guide_random_slope) [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) for each observation
    guide_random_slope ~ Normal(0, slope_noise^2) for each (guide,replicate) pair
    
    Parameters
    ----------
    Data: a seabass.Data object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """
    
    if not NT_model: 
        
        prior = get_dist("guide_score", t_df, data.device)
                
        if per_gene_variance: 
            log_guide_std_mean = convertr(log_guide_std_mean, "log_guide_std_mean", device = data.device)
            log_guide_std_std = convertr(log_guide_std_std, "log_guide_std_std", device = data.device)
            log_guide_std = pyro.sample(
                "log_guide_std",
                 dist.Normal(log_guide_std_mean, log_guide_std_std).expand([data.num_genes]).to_event(1)
            )
            guide_score = pyro.sample(
                "guide_score", 
                prior(0., log_guide_std.exp()[data.guide_to_gene]).to_event(1)
            )

        else:
            guide_std = convertr(guide_std, "guide_std", device = data.device)
            guide_score = pyro.sample(
                "guide_score", 
                prior(0., guide_std).expand([data.num_guides]).to_event(1)
            )


    if hierarchical_noise: 
        log_sigma_noise_mean = convertr(log_sigma_noise_mean, "log_sigma_noise_mean", device = data.device)
        log_sigma_noise_std = convertr(log_sigma_noise_std, "log_sigma_noise_std", device = data.device)
        log_sigma_noise = pyro.sample(
            "log_sigma_noise",
             dist.Normal(log_sigma_noise_mean, log_sigma_noise_std).expand([data.num_guides]).to_event(1)
        )
    else: 
        sigma_noise = convertr(sigma_noise, "sigma_noise", device = data.device)
    
    if hierarchical_slope: 
        log_slope_noise_mean = convertr(log_slope_noise_mean, "log_slope_noise_mean", device = data.device)
        log_slope_noise_std = convertr(log_slope_noise_std, "log_slope_noise_std", device = data.device)
        log_slope_noise = pyro.sample(
            "log_slope_noise",
             dist.Normal(log_slope_noise_mean, log_slope_noise_std).expand([data.num_guides]).to_event(1)
        )
    else:
        if slope_noise != 0.:
            slope_noise = convertr(slope_noise, "slope_noise", device = data.device)
    
    mean = torch.zeros_like(data.logFC) if NT_model else guide_score[data.guide_indices]
    
    slope_distribution = get_dist("slope", slope_t_df, data.device) 
    if data.multiday: 
        if slope_noise != 0.: 
            random_slope = pyro.sample(
                "random_slope", 
                slope_distribution(0., log_slope_noise.exp()).expand_by([data.num_replicates]).to_event(2) 
                if hierarchical_slope else
                slope_distribution(0., slope_noise).expand([data.num_replicates,data.num_guides]).to_event(2)
            )
            mean = (mean + random_slope[data.replicate,data.guide_indices]) * data.timepoint
        else: 
            mean *= data.timepoint
    
    noise_distribution = get_dist("noise", noise_t_df, data.device) 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample(
            "obs", 
            noise_distribution(
                mean, 
                log_sigma_noise.exp()[data.guide_indices] if hierarchical_noise else sigma_noise
            ), 
            obs = data.logFC)

def guide_structured(data, NT_model, init_guide_scores = None, init_random_slopes = None):
    
    assert(data.multiday) # only makes sense in this setting
    
    #P = data.num_replicates + (0 if NT_model else 1)
    
    init = torch.zeros(data.num_guides,data.num_replicates)  if init_random_slopes is None else init_random_slopes
    
    if not NT_model: 
        init = torch.cat([
            init,
            torch.zeros(data.num_guides,1) if (init_guide_scores is None) else init_guide_scores[:,None]], 
            dim = 1)
        
    P = init.shape[1]
    loc = pyro.param("loc", init)
    scale = pyro.param(
        "scale",
        torch.full_like(loc, 0.1), 
        constraint = constraints.positive
    )
    unit_tril = pyro.param(
        "unit_tril",
        eye_like(loc, P).repeat([data.num_guides,1,1]), 
        constraint = constraints.unit_lower_cholesky
    )
    
    scale_tril = scale[...,None] * unit_tril
    
    if NT_model: 
        random_slope_T = pyro.sample( # will be guides x P
            "random_slope_T", 
            dist.MultivariateNormal(loc, scale_tril = scale_tril).to_event(1), 
            infer={'is_auxiliary': True}
        )
        random_slope = pyro.sample(
            "random_slope", 
            dist.Delta(random_slope_T.T).to_event(2))
    else:
        post = pyro.sample( # will be guides x P
            "post", 
            dist.MultivariateNormal(loc, scale_tril = scale_tril).to_event(1), # probably need a to_event here?
            infer={'is_auxiliary': True}
        )
        guide_score = pyro.sample(
            "guide_score", 
            dist.Delta(post[...,-1]).to_event(1))
        random_slope = pyro.sample(
            "random_slope", 
            dist.Delta(post[...,0:-1].T).to_event(2))
    
def extract_params(
    to_optimize, 
    NT_model, 
    hierarchical_noise, 
    hierarchical_slope, 
    per_gene_variance,
    structured_guide,
    slope_noise
):

    to_return = { k:pyro.param("AutoGuideList.0." + k) for k in to_optimize }
    
    list_idx = 1
    if hierarchical_noise: 
        to_return["sigma_noise_std"] = pyro.param('AutoGuideList.1.loc').exp()
        to_return["log_sigma_noise_se"] = pyro.param('AutoGuideList.1.scale')
        list_idx += 1
    
    if hierarchical_slope: 
        to_return["slope_noise_std"] = pyro.param('AutoGuideList.%i.loc' % list_idx).exp()
        to_return["log_slope_noise_se"] = pyro.param('AutoGuideList.%i.scale' % list_idx)
        list_idx += 1
        
    if per_gene_variance: 
        to_return["guide_std"] = pyro.param('AutoGuideList.%i.loc' % list_idx).exp()
        to_return["log_guide_std_se"] = pyro.param('AutoGuideList.%i.scale' % list_idx)
        list_idx += 1
    
    if structured_guide: 
        loc = pyro.param("loc")
        #scale_tril = pyro.param("scale") * pyro.param("unit_tril").diagonal(dim1=-1,dim2=-2)
        scale_tril = pyro.param("scale")[...,None] * pyro.param("unit_tril")
        scale_tril *= scale_tril # square
        scale_tril = scale_tril.sum(-1).sqrt() 
        
        if NT_model:
            to_return["random_slope"] = loc
            to_return["random_slope_se"] = scale_tril
        else: 
            to_return["random_slope"] = loc[...,0:-1]
            to_return["random_slope_se"] : scale_tril[...,0:-1]
            to_return["guide_score"] = loc[...,-1]
            to_return["guide_score_se"] = scale_tril[...,-1]
    else: 
        if not NT_model:
            to_return["guide_score"] = pyro.param('AutoGuideList.%i.loc' % list_idx)
            to_return["guide_score_se"] = pyro.param('AutoGuideList.%i.scale' % list_idx)
            list_idx +=1 
        
        if slope_noise != 0.: 
            to_return["random_slope"] = pyro.param('AutoGuideList.%i.loc' % list_idx)
            to_return["random_slope_se"] = pyro.param('AutoGuideList.%i.scale' % list_idx)
            list_idx +=1 
        
    return { k:v.detach().cpu().numpy() for k,v in to_return.items() }

def fit(
    data,
    NT_model, 
    hierarchical_noise,
    hierarchical_slope,
    per_gene_variance = True, 
    iterations = 500,
    print_every = 200,
    end = "\r", 
    lr = 0.03,
    slope_noise = None, # set to None to learn, fix to a value (0 to have no random_slope)
    log_slope_noise_mean = None,
    log_slope_noise_std = None,
    sigma_noise = None, 
    log_sigma_noise_mean = None,
    log_sigma_noise_std = None,
    guide_std = None, 
    log_guide_std_mean = None,
    log_guide_std_std = None,
    t_df = None, # None to learn t in StudentT, 0=Laplace, 1=Cauchy, Inf=Normal. Other values given fixed t StudentT.
    noise_t_df = None,
    slope_t_df = None, 
    structured_guide = True,
    init_guide_scores = None, 
    init_random_slopes = None,
    stall_window = 10,
    max_particles = 32
): 
    
    if not init_guide_scores is None: 
        init_guide_scores = torch.tensor(init_guide_scores, dtype = data.logFC.dtype, device = data.device)

    if not init_random_slopes is None:
        init_random_slopes = torch.tensor(init_random_slopes, dtype = data.logFC.dtype, device = data.device)

    if NT_model: per_gene_variance = False

    pyro.clear_param_store()
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    zero = one*0.
    model = lambda data:  model_base(
        data, 
        NT_model = NT_model,
        hierarchical_noise = hierarchical_noise,
        hierarchical_slope = hierarchical_slope,
        per_gene_variance = per_gene_variance,
        log_sigma_noise_mean = dist.Cauchy(zero,one) if (log_sigma_noise_mean is None) else log_sigma_noise_mean, 
        log_sigma_noise_std = dist.HalfCauchy(one) if (log_sigma_noise_std is None) else log_sigma_noise_std, 
        log_slope_noise_mean = dist.Cauchy(zero,one) if (log_slope_noise_mean is None) else log_slope_noise_mean, 
        log_slope_noise_std = dist.HalfCauchy(one) if (log_slope_noise_std is None) else log_slope_noise_std, 
        log_guide_std_mean = dist.Cauchy(zero,one) if (log_guide_std_mean is None) else log_guide_std_mean, 
        log_guide_std_std = dist.HalfCauchy(one) if (log_guide_std_std is None) else log_guide_std_std, 
        sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise,
        slope_noise = dist.HalfCauchy(one) if (slope_noise is None) else slope_noise,
        guide_std = dist.HalfCauchy(one) if (guide_std is None) else guide_std ,
        t_df = dist.Gamma(two,two/10.) if (t_df is None) else t_df, 
        noise_t_df = dist.Gamma(two,two/10.) if (noise_t_df is None) else noise_t_df,
        slope_t_df = dist.Gamma(two,two/10.) if (slope_t_df is None) else slope_t_df)
    
    to_optimize = []
    
    assert(not (hierarchical_slope and slope_noise==0.)) 
    
    if slope_noise == 0.: 
        structured_guide = False # makes no difference
    
    if not NT_model: 
        if per_gene_variance: 
            if log_guide_std_mean is None: to_optimize.append("log_guide_std_mean")
            if log_guide_std_std is None: to_optimize.append("log_guide_std_std")
        else: 
            if guide_std is None: to_optimize.append("guide_std")
        if t_df is None: to_optimize.append("guide_score_t_df")

    if hierarchical_noise: 
        if log_sigma_noise_mean is None: to_optimize.append("log_sigma_noise_mean")
        if log_sigma_noise_std is None: to_optimize.append("log_sigma_noise_std")
    else: 
        if sigma_noise is None: to_optimize.append("sigma_noise")
         
    if hierarchical_slope:
        if log_slope_noise_mean is None: to_optimize.append("log_slope_noise_mean")
        if log_slope_noise_std is None: to_optimize.append("log_slope_noise_std")
    else:
        if slope_noise is None: to_optimize.append("slope_noise")
            
    if noise_t_df is None: to_optimize.append("noise_t_df")
        
    if slope_t_df is None and not slope_noise == 0.: to_optimize.append("slope_t_df")
    
    init_opt = {
        "log_guide_std_mean" : 0., 
        "log_guide_std_std" : 1., 
        "guide_std" : 1., 
        "guide_score_t_df" : 5., 
        "log_sigma_noise_mean" : 0., 
        "log_sigma_noise_std" : 1., 
        "sigma_noise" : 1., 
        "log_slope_noise_mean" : 0., 
        "log_slope_noise_std" : 1., 
        "slope_noise" : 1., 
        "noise_t_df" : 5.,
        "slope_t_df" : 5.
    }
    init_opt = {k:torch.tensor(v, device = data.device) for k,v in init_opt.items()}
    
    print("Optimizing:"," ".join(to_optimize))
    
    guide = AutoGuideList(model)
    guide.add(AutoDelta(
        poutine.block(model, expose = to_optimize),
        init_loc_fn = init_to_value(values=init_opt)
    ))

    if hierarchical_noise: # or hierarchical slope, this handles random variances
        guide.add(AutoDiagonalNormal(
            poutine.block(
                model, 
                expose = ["log_sigma_noise"]),
            init_loc_fn = init_to_value(values={"log_sigma_noise" : torch.zeros(data.num_guides,device=data.device)})))
    if hierarchical_slope: 
        guide.add(AutoDiagonalNormal(
            poutine.block(
                model, 
                expose = ["log_slope_noise"]),
            init_loc_fn = init_to_value(values={"log_slope_noise" : torch.zeros(data.num_guides,device=data.device)})))
    if per_gene_variance: 
        guide.add(AutoDiagonalNormal(
            poutine.block(
                model, 
                expose = ["log_guide_std"]),
            init_loc_fn = init_to_value(values={"log_guide_std" : torch.zeros(data.num_genes,device=data.device)})))

    if structured_guide: 
        guide.add(lambda data: guide_structured(
            data, 
            NT_model = NT_model,
            init_guide_scores = init_guide_scores, 
            init_random_slopes = init_random_slopes))
    else: 
        if not NT_model: 
            guide.add(AutoDiagonalNormal(
                poutine.block(model, expose = ["guide_score"]),
                init_loc_fn = init_to_value(
                    values={"guide_score" : torch.zeros(data.num_guides,device=data.device) if init_guide_scores is None else init_guide_scores })))
        if slope_noise != 0.: 
            guide.add(AutoDiagonalNormal(
                poutine.block(model, expose = ["random_slope"]),
                init_loc_fn = init_to_value(
                    values={"random_slope" :  torch.zeros([data.num_replicates,data.num_guides],device=data.device) if init_random_slopes is None else init_random_slopes.T })))
    
    adam = pyro.optim.Adam({"lr": lr})

    # train/fit model
    #accurate_elbo_func = Trace_ELBO(num_particles = 100, vectorize_particles = True)(model, guide)
    
    #losses = [ accurate_elbo_func(data).item() ]
    #print("Initial loss: %.4f" % (losses[0] / len(data.guide_indices)))
    losses = []
    optim_record = []
    num_particles = 1
       
    while num_particles <= max_particles: 
        svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles = num_particles, vectorize_particles = False))
        iteration = 0 
        while True: 
            loss = svi.step(data)
            losses.append(loss)
            optim_record.append({ k:pyro.param("AutoGuideList.0." + k).detach().item() for k in to_optimize })
            iteration += 1
            if iteration % print_every == 0:
                print("[iteration %04d] loss: %.4f" % (iteration + 1, loss / len(data.guide_indices)), end = end)
            if iteration > stall_window: 
                R,p = scipy.stats.pearsonr(np.arange(stall_window), losses[-stall_window:])
                if p>0.05 or R>0. or iteration > iterations: 
                    num_particles *= 2
                    print("Stalled after %i iterations. Increasing num_particles to %i." % (iteration + 1, num_particles))
                    break

    posterior = extract_params(
        to_optimize, 
        NT_model = NT_model, 
        hierarchical_noise = hierarchical_noise, 
        hierarchical_slope = hierarchical_slope,
        per_gene_variance = per_gene_variance,
        structured_guide = structured_guide, 
        slope_noise = slope_noise
    )
    
    if not NT_model: 
        posterior["z"] = posterior["guide_score"] / posterior["guide_score_se"]
        posterior["p"] = 2. * norm.cdf(-np.abs(posterior["z"]))
        posterior["sig"],posterior["q"] = statsmodels.stats.multitest.fdrcorrection(posterior["p"], alpha=0.2)

    optim_record = {key: [d[key] for d in optim_record] for key in optim_record[0]}

    return( model, guide, losses, posterior, optim_record )



