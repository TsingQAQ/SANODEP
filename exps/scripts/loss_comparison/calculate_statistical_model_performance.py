"""
Read CSV files of each random seed and calculate the mean and standard deviation of the MSE and NLL values for each model.
need to specify the model experiment directory, forcast prob and the range of random seed

Usage:
    python exps/scripts/loss_comparison/calculate_statistical_model_performance.py --exp_dir exps/experiments/lotka_voterra/loss_comparison/sanodep/cond_init_cond_dynamic --rseed_range 0 6 --eval_fc_prob 1.0 --fc_prob 0.5
"""

import os
import pandas as pd
import numpy as np
import argparse

def calculate_statistics(experiment_directory: str, random_seed_range, forecast_prob, eval_forcast_prob):
    # Initialize lists to store MSE and NLL values
    mse_values = []
    nll_values = []

    # Loop over the range of random seeds
    for seed in random_seed_range:
        # Construct the file path
        file_path = os.path.join(experiment_directory, f"forcast_prob{forecast_prob}", \
                                 f"seed_{seed}", "evaluations", f"evaluation_metrics_forcst_prob_{eval_forcast_prob}.csv")  # Adjust the file name pattern to your needs

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Calculate the MSE and NLL values and append them to the lists
        mse_values.append(df["MSE"].mean())  # Adjust the column name to your needs
        nll_values.append(df["NLL"].mean())  # Adjust the column name to your needs

    # Calculate the mean and standard deviation of the MSE and NLL values
    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    nll_mean = np.mean(nll_values)
    nll_std = np.std(nll_values)

    # Print the results
    print(f"MSE collections: {mse_values}")
    print(f"MSE: mean = {mse_mean}, std = {mse_std}")
    print(f"NLL collections: {nll_values}")
    print(f"NLL: mean = {nll_mean}, std = {nll_std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate statistics for model performance.')
    parser.add_argument('--exp_dir', type=str, required=True, help='The directory containing the experiment results.')
    parser.add_argument('--rseed_range', type=int, nargs='+', required=True, help='The range of random seeds.')
    parser.add_argument('--fc_prob', type=float, required=True, help='The forecast probability.')
    parser.add_argument('--eval_fc_prob', type=float, required=True, help='The evaluation forecast probability.')

    args = parser.parse_args()

    calculate_statistics(args.exp_dir, range(*args.rseed_range), args.fc_prob, args.eval_fc_prob)