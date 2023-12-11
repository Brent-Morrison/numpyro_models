#!/usr/bin/env bash

# https://cloud.google.com/deep-learning-vm/docs/images

# Variables
printf "\n----- SET VARIABLES -----------------------------\n\n"
PROJECT_ID=cloudstoragepythonuploadtest
INSTANCE_NAME=pytorch-vm
LOCATION=australia-southeast1
ZONE=australia-southeast1-a
BUCKET_NAME=brent_test_bucket
OBJECT_LOCATION=/c/Users/brent/Documents/R/Misc_scripts/m01_preds.csv
DELETE=false

# Set the project 
gcloud config set project ${PROJECT_ID}

# Create bucket from local development environment
printf "\n----- COPY FILES TO STORAGE\n\n"
gcloud storage buckets create gs://${BUCKET_NAME} --project=${PROJECT_ID} --location=${LOCATION}

# Upload local file to bucket - training data
gcloud storage cp ${OBJECT_LOCATION} gs://${BUCKET_NAME}/

# Create VM instance
printf "\n----- CREATE VM ---------------------------------\n\n"
gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --accelerator="type=nvidia-tesla-p4,count=1" \
    --metadata="install-nvidia-driver=True" \
    --metadata-from-file=startup-script=startup.sh
