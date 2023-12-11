#!/bin/bash

# Viewing the output of a Linux startup script
# sudo journalctl -u google-startup-scripts.service
# https://raw.githubusercontent.com/pyro-ppl/numpyro/master/examples/gp.py

# Variables
MODEL_ID=m03

# Install numpyro
pip3 install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy python script & run (TO DO - investigate exposing files via mounting with gcsfuse)
sudo chmod -R a+rwx /home/brent 
curl https://raw.githubusercontent.com/Brent-Morrison/numpyro_models/master/${MODEL_ID}.py --output /home/brent/${MODEL_ID}.py
curl https://raw.githubusercontent.com/Brent-Morrison/numpyro_models/master/${MODEL_ID}_config.json --output /home/brent/${MODEL_ID}_config.json
python3 /home/brent/${MODEL_ID}.py -c /home/brent/${MODEL_ID}_config.json