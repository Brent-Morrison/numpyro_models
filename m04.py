"""
Hierachical bayesian neural network
https://pmelchior.net/blog/bayesian-inference-three-ways.html
"""

import argparse
import time

from jax import vmap, jit
import jax.numpy as jnp
from jax.random import PRNGKey, split, normal
import jax.random as random
jnp.set_printoptions(precision=3, linewidth=125)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime as dt
import random as r
from string import ascii_lowercase
import pickle



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


def gen_stock_data(n_sectors=4, n_stocks=5, months=24):
    # Create sector and stock labels
    sect = [chr(x) for x in range(65, 65 + n_sectors)]
    sect = [item for item in sect for _ in range(n_stocks)]
    stocks = [''.join([r.choice(ascii_lowercase) for _ in range(3)]) for _ in range(n_sectors*n_stocks)]  
    df = pd.DataFrame({"sector" : sect, "stock" : stocks})
    # Create list of dates
    dates = pd.date_range(dt.datetime(2020,1,1), periods=months, freq="M").tolist()
    # Create market state by date, being sin wave of 1.5 wavelengths, eg. nil > peak > nil > trough > nil > peak > nil,
    # and cross join to stocks / sectors
    df = pd.merge(df, pd.DataFrame({"date_stamp" : dates, "mkt_state" : np.sin(np.arange(months)/(months/10))}), how="cross")
    # Create sector state, being value between 0.1 and 0.2 specific to each sector (TO DO: advanced - vary over time)
    df["sect_state0"] = np.linspace(0.1, 0.2, n_sectors).repeat(months*n_stocks)
    # Create slowly changing stock specific state (this could represent ROE, leverage, volatility, etc.)
    df["stock_state0"] = np.where(df["date_stamp"] < dates[int(months/2)], 0.1, 0.2) 
    # Stock return is function of the stock state, ie., a loading or beta on that state
    # The beta itself is a function of the market state
    df["mkt_state_diff"] = df.groupby("stock").mkt_state.diff(periods=1)
    df["mkt_state_sign"] = np.sign(df["mkt_state"])
    df["stock_state0_beta"] = np.select(
        [(df["mkt_state_diff"] > 0) & (np.sign(df["mkt_state"]) > 0),
         (df["mkt_state_diff"] > 0) & (np.sign(df["mkt_state"]) < 0),
         (df["mkt_state_diff"] < 0) & (np.sign(df["mkt_state"]) > 0),
         (df["mkt_state_diff"] < 0) & (np.sign(df["mkt_state"]) < 0)],
        [0.4, 0.8, -0.8, -0.4],
        default=0.4)
    # Average of stock return, dependent upon mkt_state, sect_state and stock_state
    df["stock_mean_rtn"] = df["stock_state0_beta"] * df["stock_state0"]
    df["stock_stdev_rtn"] = df["sect_state0"] / 2 
    df["stock_rtn"] = np.random.normal(loc=df["stock_mean_rtn"], scale=df["stock_stdev_rtn"])
    df["stock_rtn_binary"] = np.where(df["stock_rtn"] < np.median(df["stock_rtn"]), 0, 1)
    return df



# Define model -------------------------------------------------------------------------------------------------------------

def model(layer_sizes, X, Y=None):
    """X must have 3 dimension
    The first dimension is the groups, second the rows 
    and third the columns
    """
    D_C, _, D_X = X.shape
    D_Y = 1
    layer_sizes = (D_X, *layer_sizes, D_Y)

    w_mean, w_std = [], []
    for i, (D_in, D_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # Group mean distribution & standard-deviation for input to hidden layer
        w_c = numpyro.sample(f"w{i}_c", dist.Normal(jnp.zeros((D_in, D_out)), jnp.ones((D_in, D_out))))
        w_c_std = numpyro.sample(f"w{i}_c_std", dist.HalfNormal(1.0))
        # TO DO - ADD BIAS

        w_mean.append(w_c)
        w_std.append(w_c_std)

    # Data to new variable
    z = X 

    # Size is the number of groups, dim=-3 refers to left most dimension in ndim=3 array 
    with numpyro.plate("plate_i", size=D_C, dim=-3): 
        for k, (D_in, D_out, w_c, w_c_std) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], w_mean, w_std)):
            w_all = numpyro.sample(f"w{k}_all", dist.Normal(jnp.zeros((1, D_in, D_out)), jnp.ones((1, D_in, D_out))))
            w = w_all * w_c_std + w_c
            z = (jnp.tanh(jnp.matmul(z, w)) if k != len(layer_sizes) - 2 else jnp.matmul(z, w))  # output of the neural network, MAKE EXPLICIT??

    z = z.squeeze(-1) # remove the last dimension

    # Bernoulli likelihood <= Binary classification
    Y = numpyro.sample("Y", dist.Bernoulli(logits=z), obs=Y)



# Sample & save model ------------------------------------------------------------------------------------------------------

def sample(model, args, rng_key, layer_sizes, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains
    )
    
    mcmc.run(rng_key, layer_sizes, X, Y)
    mcmc.print_summary()
    post_samples = mcmc.get_samples() 

    with open("m04_post_samples.pickle", "wb") as handle:
        pickle.dump(post_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nMCMC elapsed time:", time.time() - start)


# Predict new data ---------------------------------------------------------------------------------------------------------

def predict(rng_key, layer_sizes, X):
    
    with open("m04_post_samples.pickle", "rb") as handle:
        m04_post_samples = pickle.load(handle)
    
    predictive = Predictive(model, posterior_samples=m04_post_samples)

    preds = predictive(rng_key, layer_sizes=layer_sizes, X=X)["Y"]
    means = np.mean(preds, axis=0)
    quantiles = np.percentile(preds, [5.0, 95.0], axis=0)
    
    return means, quantiles


# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    # Convert date parameter from string to date, for use as data frame filter
    #date_filter = dt.datetime.strptime(args.date_filter, '%Y-%m-%d').date()
    #date_filter = args.date_filter

    # Training data
    n_sectors = 10
    n_stocks = 100
    months = 36
    df = gen_stock_data(n_sectors, n_stocks, months)

    le = LabelEncoder()
    df['sector_tf'] = le.fit_transform(df['sector'].values)
    X_train = df[["stock_state0","stock_state1"]].values
    Y_train = df[["stock_rtn_binary"]].values


    X_train = X_train.reshape(n_sectors, X_train.shape[0]//n_sectors, -n_sectors)
    Y_train = Y_train.reshape(n_sectors, Y_train.shape[0]//n_sectors)
    X_test  = X_train

    # Prior predictive check grid
    grid = jnp.mgrid[0.1:0.2:100j, 0.1:0.2:100j].reshape((2, -1)).T
    grid_3d = jnp.repeat(grid[None, ...], n_sectors, axis=0)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    sample(model, args, rng_key, layer_sizes=[5,5], X=X_train, Y=Y_train)

    # do prediction
    means, quantiles = predict(rng_key=rng_key_predict, layer_sizes=[5,5], X=X_test)
    pd.DataFrame({'date_stamp': df["date_stamp"], 'Yhat': means.reshape(-1), 'lower': quantiles[0, :].reshape(-1), 'upper': quantiles[1, :].reshape(-1)}) \
        .to_csv('/c/Users/brent/Documents/R/Misc_scripts/m04_preds.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNN")
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
# python m04.py -d 2021-12-31 -f /c/Users/brent/Documents/R/Misc_scripts/stocks.csv


"""
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

def get_mean_predictions(predictions):
    mean_prediction = jnp.mean(predictions, axis=0)
    return mean_prediction

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
    pred_train  = get_mean_predictions(predictions)

    predictions = get_predictions(model, rng_key_test, samples, X_test  , layer_sizes, *args, **bnn_kwargs)
    pred_test   = get_mean_predictions(predictions)
    ppc_grid    = get_predictions(model, rng_key_grid, samples, grid    , layer_sizes, *args, **bnn_kwargs)
    
    return pred_train, pred_test, ppc_grid, samples


num_warmup = 1000
num_samples = 500
num_chains = 2
hidden_layers = [5, 5]

pred_train, pred_test, ppc_grid, samples = fit_and_eval(
    model=hierarchical_bnn,
    training_data=(X_train, Y_train),
    test_data=(X_test, Y_test),
    grid=grid_3d,
    layer_sizes=(X_train.shape[-1], *hidden_layers),
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
)

"""