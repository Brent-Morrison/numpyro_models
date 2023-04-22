import argparse
import os
import time

import jax
from jax import random
import jax.numpy as jnp
jnp.set_printoptions(precision=3, linewidth=125)

import numpyro
from numpyro.infer import MCMC, NUTS , Predictive
import numpyro.distributions as dist


# Market index
mkt_idx = jnp.tile(jnp.arange(1,11), 4)

def model(Y=None, X=None):

    # Ones for intercept
    X = jnp.concatenate((jnp.ones(X.shape[0])[:,None], X), axis=1)

    # Priors
    bX = numpyro.sample("bX", dist.Normal(0.0, 0.5).expand(batch_shape=[X.shape[1]])) # assumes leading column of 1's
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Mean
    mu = jnp.matmul(X,bX)

    # Likelihood
    with numpyro.plate("data", len(Y)):
        numpyro.sample("Y", dist.Normal(mu, sigma), obs=Y)


#rng_key = random.PRNGKey(0)
#mcmc = MCMC(NUTS(model), num_warmup=2000, num_samples=2000, num_chains=2)
#mcmc.run(rng_key, Y=rtn_obs, X=X)
#mcmc.print_summary()
#posterior_draws = mcmc.get_samples()

# Sample & save model ------------------------------------------------------------------------------------------------------

def sample(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning
    )
    mcmc.run(rng_key, Y, X)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def predict(mcmc, n=5):
    samples = mcmc.get_samples()
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    vmap_args = (
        random.split(rng_key_predict, samples["nu"].shape[0]),
        samples["nu"],
        samples["eta"],
    )
    preds_map = jax.vmap(
        lambda key, nu, eta: np.mean(
            nd.Beta(eta, nu - eta).sample(key, sample_shape=(n,)), axis=0
        )
    )
    preds = preds_map(*vmap_args)
    means = np.mean(preds, axis=0)
    quantiles = np.percentile(preds, [5.0, 95.0], axis=0)
    return means, quantiles