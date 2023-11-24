import argparse
import time

import jax
from jax import random
import jax.numpy as jnp
jnp.set_printoptions(precision=3, linewidth=125)

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist

import pickle

import numpy as np
import pandas as pd
import datetime as dt

# https://forum.pyro.ai/t/cant-get-numpyro-to-use-gpu-on-linux/4479

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


def make_data(samples=25, intercept=1, coefficients=[2, 3], noise=0.1):
    
    X = np.random.rand(samples, len(coefficients))
    coefficients = np.array(coefficients)
    Y = np.dot(X, coefficients) + intercept
    Y = Y + np.random.normal(scale=noise, size=Y.shape)

    idx = pd.date_range(dt.datetime(2020, 1, 1), periods=samples, freq="M").values

    return X, Y, idx


# Define model -------------------------------------------------------------------------------------------------------------

def model(X=None, Y=None):

    # Ones for intercept
    # x = x[:,None] if x.ndim == 1 else x
    # cell 7 here - https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html
    if X is not None:
        X = jnp.concatenate((jnp.ones(X.shape[0])[:,None], X), axis=1) 

    # Priors
    if X is not None:
        bX = numpyro.sample("bX", dist.Normal(0.0, 0.5).expand(batch_shape=[X.shape[1]])) # assumes leading column of 1's
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Mean
    if X is not None:
        mu = jnp.matmul(X, bX)
    else:
        mu = 0.0

    # Likelihood
    with numpyro.plate("data", len(X)): #len(Y)):
        numpyro.sample("Y", dist.Normal(mu, sigma), obs=Y)



# Sample & save model ------------------------------------------------------------------------------------------------------

def sample(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains
    )
    
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    post_samples = mcmc.get_samples() 

    with open("m01_post_samples.pickle", "wb") as handle:
        pickle.dump(post_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nMCMC elapsed time:", time.time() - start)



# Predict new data ---------------------------------------------------------------------------------------------------------

def predict(data, rng_key):
    
    with open("m01_post_samples.pickle", "rb") as handle:
        m01_post_samples = pickle.load(handle)
    
    predictive = Predictive(model, posterior_samples=m01_post_samples)

    #preds = predictive(rng_key, Y=data)["Y"]
    preds = predictive(rng_key, X=data, Y=None)["Y"]
    means = np.mean(preds, axis=0)
    quantiles = np.percentile(preds, [5.0, 95.0], axis=0)
    
    return means, quantiles


# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    # Convert date parameter from string to date, for use as data frame filter
    #date_filter = dt.datetime.strptime(args.date_filter, '%Y-%m-%d').date()
    #date_filter = args.date_filter

    # Training data
    if args.test:
        X, Y, idx = make_data(coefficients=[2,3])
    else:
        X, Y, idx = get_data(filepath=args.filepath, date_filter=args.date_filter)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    sample(model, args, rng_key, X=X, Y=Y)

    # do prediction
    means, quantiles = predict(data=X, rng_key=rng_key_predict)  # TO DO - this Y should be new data
    print("Mean", means.shape)
    print("Quantile", quantiles.shape)
    print("Index", idx.shape)
    pd.DataFrame({'date_stamp': idx, 'Yhat': means, 'lower': quantiles[0, :], 'upper': quantiles[1, :]}) \
        .to_csv('/c/Users/brent/Documents/R/Misc_scripts/m01_preds.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression")
    parser.add_argument("-t", "--test"       , nargs="?", default=False, type=bool)
    parser.add_argument("-d", "--date_filter", nargs="?", default="2022-12-31", type=str)
    parser.add_argument("-f", "--filepath"   , nargs="?", default="/c/Users/brent/Documents/R/Misc_scripts/stocks.csv", type=str)
    parser.add_argument("-w", "--num_warmup" , nargs="?", default=1000, type=int)
    parser.add_argument("-s", "--num_samples", nargs="?", default=2000, type=int)
    parser.add_argument("-c", "--num_chains" , nargs="?", default=2   , type=int)

    args = parser.parse_args()

    #numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)


# conda activate pytorch_pyro
# cd ~/numpyro_models/numpyro_models
# python m01.py -d 2021-12-31 -f /c/Users/brent/Documents/R/Misc_scripts/stocks.csv
# python m01.py -t True
# conda activate pytorch_pyro && python ~/numpyro_models/numpyro_models/m01.py