#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logging.info('Downloading artifacts')
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info("Reading data into dataframe")
    df = pd.read_csv(artifact_path)

    logger.info("Beginning to clean data")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Finished cleaning data")

    logger.info("Create clean data csv file")
    df.to_csv(args.output_artifact, index=False)

    logger.info("Create artifact for W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )


    logger.info("Log artifact to W&B")
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info("remove local csv file")
    os.remove(args.output_artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='Name of artifact within wandb',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='Name of artifact to be logged to wandb',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='Type of artifact to be created within wandb',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='Description of artifact being logged to wandb',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='Minimum price for filtering outliers',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='Maximum price for filtering outliers',
        required=True
    )


    args = parser.parse_args()

    go(args)
