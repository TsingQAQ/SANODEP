"""
Read all csv files in related folders and calculate the mean and standard deviation of the MSE and NLL values for each model.
Generate the latex table for the mean and standard deviation of the MSE and NLL values for each model.
Usage:
# 1. compare NODEP and SANODEP trained with evaluation forecast probability 0.0
# 2. compare all models with evaluation forecast probability 0.5

# python exps/scripts/model_comparison/calculate_statistical_model_performance.py --exp_dir exps/experiments/lotka_voterra/model_comparison/sanodep/ --rseed_range 0 6 --eval_fc_prob 0.5 --fc_prob 0.5
"""


import os
import pandas as pd
import numpy as np

stability_quantile = 0.95


def calculate_statistics(experiment_directory: str, random_seed_range, forecast_prob, eval_forecast_prob, ctx_traj_range, remove_invalid=False):
    # Initialize lists to store MSE and NLL values
    mse_values = []
    nll_values = []

    # Loop over the range of random seeds
    for seed in random_seed_range:
        # Construct the file path
        file_path = os.path.join(experiment_directory, f"forcast_prob{forecast_prob}", \
                                 f"seed_{seed}", "evaluations", F"ctx_traj_size_{ctx_traj_range}", f"evaluation_metrics_forcst_prob_{eval_forecast_prob}.csv")  # Adjust the file name pattern to your needs

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Calculate the MSE and NLL values and append them to the lists
        mse_values.append(df[df["MSE"] <= df["MSE"].quantile(stability_quantile)]["MSE"].mean())  # Adjust the column name to your needs
        nll_values.append(df[df["NLL"] <= df["NLL"].quantile(stability_quantile)]["NLL"].mean())  # Adjust the column name to your needs

    # Calculate the mean and standard deviation of the MSE and NLL values
    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    nll_mean = np.mean(nll_values)
    nll_std = np.std(nll_values)

    # Print the results
    # print(f"MSE collections: {mse_values}")
    # print(f"MSE: mean = {mse_mean}, std = {mse_std}")
    # print(f"NLL collections: {nll_values}")
    # print(f"NLL: mean = {nll_mean}, std = {nll_std}")
    return mse_mean, mse_std, nll_mean, nll_std


def generate_latex_table_for_NODEP_SANODEP_comparison(models_name, 
                                              exp_problem_names, 
                                              exp_dirs, 
                                              rseed_ranges, 
                                              ctx_traj_range, 
                                              dir_models_name, 
                                              filename_prefix="model_comparison", 
                                              metric='mse', 
                                              eval_forecast_prob=0.0,
                                              scaling: float = 1e2):
    """
    Generates a LaTeX table for model comparison and writes it to a .tex file.

    :param models_name: List of model names.
    :param exp_problem_names: List of experiment problem names.
    :param exp_dirs: Dictionary mapping experiment problem names to directories.
    :param rseed_ranges: Range of random seeds.
    :param ctx_traj_range: Context trajectory range.
    :param dir_models_name: Dictionary mapping model names to directory names.
    :param filename_prefix: Prefix for the output filename.
    :param metric: Performance metric ('mse' or 'nll').
    :param scaling: Scaling factor applied to the performance metrics.
    """
    def get_latex_scaling_factor(scaling):
        if scaling == 1:
            return ''
        else:
            scaling_in_caption = 1 / scaling
            exponent = int(np.log10(scaling_in_caption))
            mantissa = scaling_in_caption / (10 ** exponent)
            if np.isclose(mantissa, 1.0):
                return f"$\\times 10^{{{exponent}}}$"
            else:
                return f"$\\times {mantissa:.1f} \\times 10^{{{exponent}}}$"

        
    model_performance = {}
    for model_name in models_name:
        if '0.5' in model_name:
            trained_forecast_prob = 0.5
        elif 'GP' in model_name:
            trained_forecast_prob = np.nan
        else:
            trained_forecast_prob = 0.0
        model_performance[model_name] = {}
        for exp_problem_name in exp_problem_names:
            exp_dir = exp_dirs[exp_problem_name]
            metric_mean, mse_std, nll_mean, nll_std = calculate_statistics(
                os.path.join(exp_dir, dir_models_name[model_name]), 
                rseed_ranges, 
                trained_forecast_prob, 
                eval_forecast_prob, 
                ctx_traj_range, 
                remove_invalid=True)
            model_performance[model_name][exp_problem_name] = {}
            model_performance[model_name][exp_problem_name]['mse_mean'] = metric_mean
            model_performance[model_name][exp_problem_name]['mse_std'] = mse_std
            model_performance[model_name][exp_problem_name]['nll_mean'] = nll_mean
            model_performance[model_name][exp_problem_name]['nll_std'] = nll_std
    
    # Find the lowest mean for each problem
    lowest_means = {problem: float('inf') for problem in exp_problem_names}

    for performances in model_performance.values():
        for problem, performance in performances.items():
            mean = performance[f'{metric}_mean'] * scaling
            if mean < lowest_means[problem]:
                lowest_means[problem] = mean

    scaling_factor_in_caption = get_latex_scaling_factor(scaling)
    metric_name = 'mean-squared-error' if metric == 'mse' else 'negative log-likelihood'

    table_header = rf"""
    \begin{{table}}[h]
    \captionsetup{{font=scriptsize}}
    \caption{{Model comparison on \textbf{{interpolation}} tasks ({metric_name} {scaling_factor_in_caption}) for a range of dynamical systems.}}
    \centering
    \begin{{adjustbox}}{{max width=\textwidth}}
    \begin{{tabular}}{{lcccccc}}
      \toprule
      Models & Lotka-Volterra ($2d$) & Brusselator ($2d$) & Selkov ($2d$) & SIR ($3d$) & Lotka-Volterra ($3d$) & SIRD ($4d$) \\
      \midrule
    """

    table_footer = rf"""
      \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \label{{{{tab:{metric}_interp_comparison_nodep_sanodep}}}}
    \end{{table}}
    """
    def format_number(num):
        formatted_num = f"{num:.3g}"
        # Ensure that .0 is preserved for numbers like 22.0
        if '.' not in formatted_num and 'e' not in formatted_num:
            formatted_num = f"{num:.1f}"
        return formatted_num

    table_content = ""
    for model, performances in model_performance.items():
        row = f"      {model} & "
        for problem, performance in performances.items():
            metric_mean = performance[f'{metric}_mean'] * scaling
            metric_std = performance[f'{metric}_std'] * scaling
            formatted_mean = format_number(metric_mean)
            formatted_std = format_number(metric_std)
            if metric_mean == lowest_means[problem]:
                row += f"$\\boldsymbol{{{formatted_mean} \\pm {formatted_std}}}$ & "
            else:
                row += f"${formatted_mean} \\pm {formatted_std}$ & "
        table_content += row.rstrip(" & ") + " \\\\\n"
    
    latex_table = table_header + table_content + table_footer

    # Get the current script directory
    exps_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Join the current script directory with the filename
    file_path = os.path.join(exps_dir, 'experiments/figs', ''.join([filename_prefix, f'_{metric}.tex']))

    with open(file_path, "w") as file:
        file.write(latex_table)

    # Generate NLL table
    if metric == 'mse':
        # Generate NLL table by calling the function recursively with metric='nll' and appropriate scaling
        generate_latex_table_for_NODEP_SANODEP_comparison(
            models_name, 
            exp_problem_names, 
            exp_dirs, 
            rseed_ranges, 
            ctx_traj_range, 
            dir_models_name, 
            filename_prefix=filename_prefix, 
            metric='nll', 
            scaling=1  # Adjust scaling if necessary
        )


def calculate_statistics_for_all_models(experiment_directory: str, random_seed_range, forecast_prob, eval_forecast_prob, ctx_traj_size_range, remove_invalid=False):
    # Initialize lists to store MSE and NLL values
    mse_values = []
    nll_values = []


    # Loop over the range of random seeds
    for seed in random_seed_range:
        mse_values_per_seed = []
        nll_values_per_seed = []
        # For each seed, we loop over ctx_traj_size_range
        for ctx_traj_size in ctx_traj_size_range:
            # Construct the file path
            ctx_traj_size_folder = f"ctx_traj_size_{(ctx_traj_size, ctx_traj_size)}"
            file_path = os.path.join(experiment_directory, f"forcast_prob{forecast_prob}", \
                                     f"seed_{seed}", "evaluations", ctx_traj_size_folder, f"evaluation_metrics_forcst_prob_{eval_forecast_prob}.csv")
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            # Read the CSV file
            df = pd.read_csv(file_path)
            df = df.dropna()
            # Calculate the MSE and NLL values and append them to the lists
            mse_values_per_seed.append(df[df["MSE"] <= df["MSE"].quantile(stability_quantile)]["MSE"].mean())
            nll_values_per_seed.append(df[df["NLL"] <= df["NLL"].quantile(stability_quantile)]["NLL"].mean())
        # Append the mean of MSE and NLL values for each seed
        mse_values.append(np.mean(mse_values_per_seed))
        nll_values.append(np.mean(nll_values_per_seed))

    # Calculate the mean and standard deviation of the MSE and NLL values
    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    nll_mean = np.mean(nll_values)
    nll_std = np.std(nll_values)

    # Print the results
    # print(f"MSE collections: {mse_values}")
    print(f"MSE: mean = {mse_mean}, std = {mse_std}")
    # print(f"NLL collections: {nll_values}")
    print(f"NLL: mean = {nll_mean}, std = {nll_std}")
    return mse_mean, mse_std, nll_mean, nll_std

    
def plot_latex_table_for_all_model_comparison(models_name, 
                                              exp_problem_names, 
                                              exp_dirs, 
                                              rseed_ranges, 
                                              ctx_traj_size_range, 
                                              dir_models_name, 
                                              filename_prefix="model_comparison", 
                                              metric='mse', 
                                              scaling: float = 1e2):
    """
    Generate a LaTeX table comparing all models for forecast probabilities 1.0 and 0.0.
    """
    def get_latex_scaling_factor(scaling):
        if scaling == 1:
            return ''
        else:
            scaling_in_caption = 1 / scaling
            exponent = int(np.log10(scaling_in_caption))
            mantissa = scaling_in_caption / (10 ** exponent)
            if np.isclose(mantissa, 1.0):
                return f"$\\times 10^{{{exponent}}}$"
            else:
                return f"$\\times {mantissa:.1f} \\times 10^{{{exponent}}}$"

    def format_number(num):
        formatted_num = f"{num:.3g}"
        # Ensure that .0 is preserved for numbers like 22.0
        if '.' not in formatted_num and 'e' not in formatted_num:
            formatted_num = f"{num:.1f}"
        return formatted_num

    # Initialize a dictionary to store the performance metrics
    model_performance = {}
    forecast_probs = [1.0, 0.0]
    for forecast_prob in forecast_probs:
        model_performance[forecast_prob] = {}
        for model_name in models_name:
            if '0.5' in model_name:
                trained_forecast_prob = 0.5
            elif 'GP' in model_name:
                trained_forecast_prob = np.nan
            else:
                trained_forecast_prob = 0.0
            model_performance[forecast_prob][model_name] = {}
            for exp_problem_name in exp_problem_names:
                exp_dir = exp_dirs[exp_problem_name]
                # Call calculate_statistics_for_all_models
                metric_mean, mse_std, nll_mean, nll_std = calculate_statistics_for_all_models(
                    os.path.join(exp_dir, dir_models_name[model_name]), 
                    rseed_ranges, 
                    trained_forecast_prob, 
                    forecast_prob,  # use forecast_prob as eval_forecast_prob
                    np.arange(*ctx_traj_size_range), 
                    remove_invalid=True)
                model_performance[forecast_prob][model_name][exp_problem_name] = {}
                model_performance[forecast_prob][model_name][exp_problem_name]['mse_mean'] = metric_mean
                model_performance[forecast_prob][model_name][exp_problem_name]['mse_std'] = mse_std
                model_performance[forecast_prob][model_name][exp_problem_name]['nll_mean'] = nll_mean
                model_performance[forecast_prob][model_name][exp_problem_name]['nll_std'] = nll_std

    # Now, generate the LaTeX table
    # For each forecast_prob, find the lowest means
    lowest_means = {}
    for forecast_prob in forecast_probs:
        lowest_means[forecast_prob] = {problem: float('inf') for problem in exp_problem_names}
        for model_name, performances in model_performance[forecast_prob].items():
            for problem, performance in performances.items():
                mean = performance[f'{metric}_mean'] * scaling
                if mean < lowest_means[forecast_prob][problem]:
                    lowest_means[forecast_prob][problem] = mean

    # Now, create the LaTeX table
    scaling_factor_in_caption = get_latex_scaling_factor(scaling)
    metric_name = 'mean-squared-error' if metric == 'mse' else 'negative log-likelihood'

    table_header = rf"""
    \begin{{table}}[h]
    \captionsetup{{font=scriptsize}}
    \caption{{Model comparison ({metric_name} {scaling_factor_in_caption}) for a range of dynamical systems.}}
    \centering
    \begin{{adjustbox}}{{max width=\textwidth}}
    \begin{{tabular}}{{l{'c'*len(exp_problem_names)}}}
      \toprule
      Models & {' & '.join(exp_problem_names)} \\
      \midrule
    """

    table_footer = rf"""
      \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \label{{{{tab:{metric}_all_model_comparison}}}}
    \end{{table}}
    """

    def format_table_section(forecast_prob):
        section_header = rf"""
        \multicolumn{{{len(exp_problem_names)+1}}}{{c}}{{\textbf{{Evaluation Forecast Prob = {forecast_prob}}}}} \\
        \midrule
        """
        table_content = ""
        for model in models_name:
            performances = model_performance[forecast_prob][model]
            row = f"      {model} & "
            for problem in exp_problem_names:
                performance = performances[problem]
                metric_mean = performance[f'{metric}_mean'] * scaling
                metric_std = performance[f'{metric}_std'] * scaling
                formatted_mean = format_number(metric_mean)
                formatted_std = format_number(metric_std)
                if np.isclose(metric_mean, lowest_means[forecast_prob][problem]):
                    row += f"$\\boldsymbol{{{formatted_mean} \\pm {formatted_std}}}$ & "
                else:
                    row += f"${formatted_mean} \\pm {formatted_std}$ & "
            table_content += row.rstrip(" & ") + " \\\\\n"
        return section_header + table_content

    # Generate table sections
    table_sections = ""
    for forecast_prob in forecast_probs:
        table_sections += format_table_section(forecast_prob)

    # Combine all parts
    latex_table = table_header + table_sections + table_footer

    # Get the current script directory
    exps_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Join the current script directory with the filename
    file_path = os.path.join(exps_dir, 'experiments/figs', ''.join([filename_prefix, f'_{metric}.tex']))

    with open(file_path, "w") as file:
        file.write(latex_table)

    # Generate NLL table
    if metric == 'mse':
        # Generate NLL table by calling the function recursively with metric='nll' and appropriate scaling
        plot_latex_table_for_all_model_comparison(
            models_name, 
            exp_problem_names, 
            exp_dirs, 
            rseed_ranges, 
            ctx_traj_size_range, 
            dir_models_name, 
            filename_prefix=filename_prefix, 
            metric='nll', 
            scaling=1E-2  # Adjust scaling if necessary
        )


def generate_latex_table_for_all_models_comparison(models_name, 
                                                  exp_problem_names, 
                                                  exp_dirs, 
                                                  rseed_ranges, 
                                                  ctx_traj_size, 
                                                  dir_models_name, 
                                                  filename_prefix="all_model_comparison", 
                                                  metric='mse', 
                                                  scaling: float = 1e2):
    """
    Generates a LaTeX table comparing all models across different forecast probabilities.

    The table is divided into two sections:
    1. Forecasting (forecast_prob = 1.0)
    2. Interpolating (forecast_prob = 0.0)

    :param models_name: List of model names.
    :param exp_problem_names: List of experiment problem names.
    :param exp_dirs: Dictionary mapping experiment problem names to directories.
    :param rseed_ranges: Iterable of random seed integers.
    :param ctx_traj_size: Context trajectory size (tuple).
    :param dir_models_name: Dictionary mapping model names to directory names.
    :param filename_prefix: Prefix for the output filename.
    :param metric: Performance metric ('mse' or 'nll').
    :param scaling: Scaling factor applied to the performance metrics.
    """
    def get_latex_scaling_factor(scaling):
        if scaling == 1:
            return ''
        else:
            scaling_in_caption = 1 / scaling
            exponent = int(np.log10(scaling_in_caption))
            mantissa = scaling_in_caption / (10 ** exponent)
            if np.isclose(mantissa, 1.0):
                return f"$\\times 10^{{{exponent}}}$"
            else:
                return f"$\\times {mantissa:.1f} \\times 10^{{{exponent}}}$"

    def format_number(num):
        if np.isnan(num):
            return "--"
        formatted_num = f"{num:.3g}"
        # Ensure that .0 is preserved for numbers like 22.0
        if '.' not in formatted_num and 'e' not in formatted_num:
            formatted_num = f"{num:.1f}"
        if np.abs(num) > 1000:
            formatted_num = f"{num:.0f}"
        return formatted_num

    # Define forecast probabilities
    forecast_probs = [1.0, 0.0]
    forecast_labels = {1.0: "Forecasting", 0.0: "Interpolating"}

    # Initialize a nested dictionary to store performance metrics
    model_performance = {fp: {model: {} for model in models_name} for fp in forecast_probs}

    # Collect performance metrics
    for fp in forecast_probs:
        for model_name in models_name:
            trained_forecast_prob = 0.0  # Default
            if '0.5' in model_name:
                trained_forecast_prob = 0.5
            elif 'GP' in model_name:
                trained_forecast_prob = np.nan  # Assuming GP doesn't use forecast_prob

            for exp_problem_name in exp_problem_names:
                exp_dir = exp_dirs[exp_problem_name]
                


                # Calculate statistics
                mse_mean, mse_std, nll_mean, nll_std = calculate_statistics(
                    experiment_directory=os.path.join(exp_dir, dir_models_name[model_name]),
                    random_seed_range=rseed_ranges,
                    forecast_prob=trained_forecast_prob,
                    eval_forecast_prob=fp,
                    ctx_traj_range=ctx_traj_size,
                    remove_invalid=True
                )

                # Store metrics based on the chosen metric
                if metric == 'mse':
                    mean = mse_mean * scaling
                    std = mse_std * scaling
                elif metric == 'nll':
                    mean = nll_mean * scaling
                    std = nll_std * scaling
                else:
                    raise ValueError("Unsupported metric. Choose 'mse' or 'nll'.")

                model_performance[fp][model_name][exp_problem_name] = (mean, std)

    # Determine the lowest mean for each problem and forecast probability
    lowest_means = {fp: {problem: float('inf') for problem in exp_problem_names} for fp in forecast_probs}

    for fp in forecast_probs:
        for model, performances in model_performance[fp].items():
            for problem, (mean, _) in performances.items():
                if mean < lowest_means[fp][problem]:
                    lowest_means[fp][problem] = mean

    # Prepare LaTeX table components
    scaling_factor_in_caption = get_latex_scaling_factor(scaling)
    metric_name = 'Mean Squared Error' if metric == 'mse' else 'Negative Log-Likelihood'

    table_header = rf"""
\begin{{table}}[h]
\captionsetup{{font=scriptsize}}
\caption{{Model comparison on \textbf{{interpolation}} and \textbf{{forecasting}} tasks ({metric_name} {scaling_factor_in_caption}) for a range of dynamical systems.}}
\centering
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{l{'c'*len(exp_problem_names)}}}
  \toprule
  Models & {' & '.join(exp_problem_names)} \\
  \midrule
"""

    table_footer = rf"""
  \bottomrule
\end{{tabular}}
\end{{adjustbox}}
\label{{tab:{metric}_all_models_comparison}}
\end{{table}}
"""

    # Function to create table sections
    def create_table_section(fp):
        section_header = rf"\multicolumn{{{len(exp_problem_names)+1}}}{{c}}{{\textbf{{{forecast_labels[fp]}}}}} \\ \midrule" + "\n"
        table_content = ""
        for model in models_name:
            row = f"  {model} & "
            for problem in exp_problem_names:
                mean, std = model_performance[fp][model].get(problem, (np.nan, np.nan))
                formatted_mean = format_number(mean)
                formatted_std = format_number(std)
                if np.isclose(mean, lowest_means[fp][problem]):
                    cell = f"$\\boldsymbol{{{formatted_mean} \\pm {formatted_std}}}$"
                else:
                    cell = f"${formatted_mean} \\pm {formatted_std}$"
                row += f"{cell} & "
            table_content += row.rstrip(" & ") + " \\\\\n"
        if 'Interpolating' in section_header:
            # add a line before section header
            section_header = f"\\midrule\n{section_header}"
        return section_header + table_content

    # Generate table sections for each forecast probability
    table_sections = ""
    for fp in forecast_probs:
        table_sections += create_table_section(fp)

    # Combine all parts to form the complete LaTeX table
    latex_table = table_header + table_sections + table_footer

    # Define the output path
    # Assuming this script is located three directories deep from the 'experiments' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exps_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    output_dir = os.path.join(exps_dir, 'exps/experiments', 'figs')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(output_dir, f"{filename_prefix}_{metric}.tex")

    # Write the LaTeX table to the file
    with open(file_path, "w") as file:
        file.write(latex_table)

    print(f"LaTeX table successfully written to {file_path}")

    # Optionally, generate NLL table if the current metric is MSE
    if metric == 'mse':
        generate_latex_table_for_all_models_comparison(
            models_name=models_name,
            exp_problem_names=exp_problem_names,
            exp_dirs=exp_dirs,
            rseed_ranges=rseed_ranges,
            ctx_traj_size=ctx_traj_size,
            dir_models_name=dir_models_name,
            filename_prefix=filename_prefix,
            metric='nll',
            scaling=1e-2  # Adjust scaling for NLL if necessary
        )

if __name__ == "__main__":
    exp_problem_names = ['Lotka-Volterra (2d)', 'Brusselator (2d)', 'Selkov (2d)', 'SIR (3d)', 'Lotka-Volterra (3d)', 'SIRD (4d)']
    exp_dirs = {
        'Lotka-Volterra (2d)': 'exps/experiments/lotka_voterra/model_comparison/', 
        'Brusselator (2d)': 'exps/experiments/brusselator/',
        'Selkov (2d)': 'exps/experiments/selkov/',
        'SIR (3d)': 'exps/experiments/sir_unnormalized/',
        'Lotka-Volterra (3d)': 'exps/experiments/lotka_voterra_3d/',
        'SIRD (4d)': 'exps/experiments/sird/'
    }
    dir_models_name = {
        "NODEP-$\\lambda=0$": "nodep",
        "SANODEP-$\\lambda=0$": "sanodep",
        "SANODEP-$\\lambda=0.5$": "sanodep",
        "NP-$\\lambda=0.5$": "np",
        "NP-$\\lambda=0.0$": "np",
        "GP": "gp",
    }
    rseed_ranges = np.arange(5)
    # Plot of NODEP comparing with SANODEP on evaluation forecast probability 0.0
    models_name = ["NODEP-$\\lambda=0$", "SANODEP-$\\lambda=0$"]
    generate_latex_table_for_NODEP_SANODEP_comparison(models_name, exp_problem_names, exp_dirs, rseed_ranges, (0, 10), dir_models_name = dir_models_name , metric='mse', filename_prefix="model_comparison")
    generate_latex_table_for_NODEP_SANODEP_comparison(models_name, exp_problem_names, exp_dirs, rseed_ranges, (0, 10), dir_models_name = dir_models_name , metric='nll', filename_prefix="model_comparison", scaling=1e-2)
    
    # Plot of all models comparing on evaluation forecast probability 0.0
    models_name = ["NODEP-$\\lambda=0$", "NP-$\\lambda=0.0$", "NP-$\\lambda=0.5$", "SANODEP-$\\lambda=0$", "SANODEP-$\\lambda=0.5$"] # , GP"]
    generate_latex_table_for_all_models_comparison(models_name, exp_problem_names, exp_dirs, rseed_ranges, (0, 10), dir_models_name = dir_models_name , metric='mse', filename_prefix="all_model_comparison", scaling=1e2)
    
    # Plot of all models comparing on evaluation forecast probability 1.0

