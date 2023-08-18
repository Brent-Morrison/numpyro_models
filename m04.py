import argparse
import time

from jax import vmap, jit
import jax.numpy as jnp
from jax.random import PRNGKey, split, normal
import jax.random as random
jnp.set_printoptions(precision=3, linewidth=125)

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import pickle

import numpy as np
import pandas as pd



# Get data -----------------------------------------------------------------------------------------------------------------

def get_data(filepath, date_filter):
    
    # CSV
    d = pd.read_csv(filepath, index_col=None)
    d['date_stamp'] = pd.to_datetime(d['date_stamp'], format="%d/%m/%Y")
    
    # Date filter
    min_date = date_filter
    #min_date = dt.datetime(2020,6,30)
    max_date = d['date_stamp'].max()

    idx = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)]['date_stamp'].values
    Y = d.query('(date_stamp > @min_date) & (date_stamp < @max_date)')['fwd_rtn'].values
    X = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)].iloc[:, 2:4].values

    return X, Y, idx



# Define model -------------------------------------------------------------------------------------------------------------

def hierarchical_bnn(X, Y, layer_sizes):
    D_C, _, D_X = X.shape
    D_Y = 1
    layer_sizes = (D_X, *layer_sizes, D_Y)
    z = X

    w_mean, w_std = [], []
    for i, (D_in, D_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # Group mean distribution for input to hidden layer
        w_c = numpyro.sample(f"w{i}_c", dist.Normal(jnp.zeros((D_in, D_out)), jnp.ones((D_in, D_out))))
        # Group standard-deviation
        w_c_std = numpyro.sample(f"w{i}_c_std", dist.HalfNormal(1.0))

        w_mean.append(w_c)
        w_std.append(w_c_std)

    with numpyro.plate("plate_i", D_C, dim=-3):
        for k, (D_in, D_out, w_c, w_c_std) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], w_mean, w_std)):
            w_all = numpyro.sample(f"w{k}_all", dist.Normal(jnp.zeros((1, D_in, D_out)), jnp.ones((1, D_in, D_out))))
            w = w_all * w_c_std + w_c
            z = (
                jnp.tanh(jnp.matmul(z, w)) if k != len(layer_sizes) - 2 else jnp.matmul(z, w)
            )  # output of the neural network

    z = z.squeeze(-1)
    # Bernoulli likelihood <= Binary classification
    Y = numpyro.sample("Y", dist.Bernoulli(logits=z), obs=Y)



# Sample & save model ------------------------------------------------------------------------------------------------------

def run_inference(model, rng_key, num_warmup=100, num_samples=100, num_chains=1, **kwargs):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False)
    mcmc.run(rng_key, **kwargs)
    return mcmc


def get_predictions(model, rng_key, samples, X, layer_sizes, num_warmup=100, num_samples=100, num_chains=1, **bnn_kwargs):
    # helper function for prediction
    @jit
    def predict(samples, rng_key):
        model_ = handlers.substitute(handlers.seed(model, rng_key), samples)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = (
            handlers.trace(model_).get_trace(X=X, Y=None, layer_sizes=layer_sizes, **bnn_kwargs)
            if bnn_kwargs
            else handlers.trace(model_).get_trace(X=X, Y=None, layer_sizes=layer_sizes)
        )
        return model_trace["Y"]["value"]

    # predict Y at inputs X
    keys = random.split(rng_key, num_samples * num_chains)
    predictions = vmap(predict, in_axes=(0, 0))(samples, keys)
    return predictions


def fit_and_eval(
    model, 
    training_data, 
    test_data, 
    grid, 
    layer_sizes, 
    num_warmup=100, 
    num_samples=100, 
    num_chains=1, 
    **bnn_kwargs):
  
    X_train, Y_train = training_data
    X_test, Y_test = test_data

    args = [num_warmup, num_samples, num_chains]

    kwargs = {"X": X_train, "Y": Y_train, "layer_sizes": layer_sizes}
    if bnn_kwargs:
        kwargs = {**kwargs, **bnn_kwargs}

    # do inference
    rng_key, rng_key_train, rng_key_test, rng_key_grid = random.split(random.PRNGKey(0), 4)
    mcmc = run_inference(model, rng_key, *args, **kwargs)
    samples = mcmc.get_samples()

    # predict Y_train and Y_test at inputs X_traind and X_test, respectively
    predictions = get_predictions(model, rng_key_train, samples, X_train, layer_sizes, *args, **bnn_kwargs)
    pred_train = get_mean_predictions(predictions)

    predictions = get_predictions(model, rng_key_test, samples, X_test, layer_sizes, *args, **bnn_kwargs)
    pred_test = get_mean_predictions(predictions)

    ppc_grid = get_predictions(model, rng_key_grid, samples, grid, layer_sizes, *args, **bnn_kwargs)
    
    return pred_train, pred_test, ppc_grid, samples