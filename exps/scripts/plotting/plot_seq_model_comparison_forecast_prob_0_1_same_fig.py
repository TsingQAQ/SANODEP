"""
Sequentially plot the model comparison result
"""
from matplotlib import pyplot as plt
from jax import numpy as np
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter, LogFormatterSciNotation
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator
import matplotlib.ticker as ticker
from collections import OrderedDict

test_traj_num = 10
stability_quantile = 0.95


def read_batch_gp_csv_file_from_path(file_path, sys_id_range = 50):
    res_mse = []
    res_nll = []
    for sys_id in range(sys_id_range):
        try:
            df = pd.read_csv(file_path + f"evaluation_metrics_sys_id_{sys_id}_forcst_prob_1.0.csv")
            df = df.dropna()
            if len(df['mse'].to_numpy()) == 100:
                res_mse.append(df['mse'].to_numpy())
            if len(df['nll'].to_numpy()) == 100:
                res_nll.append(df['nll'].to_numpy())
        except:
            pass
    
    return np.asarray(res_mse), np.asarray(res_nll)


def plot_interpolating_and_forecasting_for_models():
    color_list_map = {'gp': 'tab:blue', 'np_0.0': 'tab:orange', 'np_0.5': 'tab:brown', 'nodep_0.0': 'tab:green', 'sanodep_0.0': 'tab:pink', 'sanodep_0.5': 'tab:purple', 
                      'grey_box_LV_sanodep': 'tab:red', 'sanodep_without_init_aug_0.0': 'tab:olive', 'sanodep_without_init_aug_0.5': 'tab:cyan'}
    pb_list = ['LV', 'Brusselator', 'Selkov', 'SIR', 'LV3D', 'SIRD']
    problem_name_mapping = {'LV': 'lotka_voterra', 'Brusselator': 'brusselator', 'FitzHug': 'FitzHugh_Nagumo', 'Selkov': 'selkov', 'SIR': 'sir_unnormalized', 'LV3D': 'lotka_voterra_3d', 'SIRD':'sird'}
    problem_title = {'LV': 'Lotka-Voterra ($2d$)', 'Brusselator': 'Brusselator ($2d$)', 'FitzHug': 'FitzHugh-Nagumo ($2d$)', 'Selkov': 'Selkov ($2d$)', 'SIR': 'SIR  ($3d$)', 'LV3D': 'Lotka-Voterra ($3d$)', 'SIRD':'SIRD ($4d$)'}
    model_list = ['sanodep_without_init_aug_0.5', 'np_0.0', 'np_0.5', 'nodep_0.0', 'sanodep_0.0', 'sanodep_0.5']#  , 'gp']
    model = {'np_0.0': 'np', 'np_0.5': 'np', 'nodep_0.0': 'nodep', 'sanodep_0.5': 'sanodep', 'sanodep_0.0': 'sanodep', 'gp':'gp', 
             'sanodep_without_init_aug_0.0': 'sanodep_without_init_cond_encode', 
              'sanodep_without_init_aug_0.5': 'sanodep_without_init_cond_encode', 
              'gp': 'gp'}
    model_label_name = {'np_0.0': 'NP-$\lambda=0$', 'np_0.5': 'NP-$\lambda=0.5$', 'nodep_0.0': 'NODEP', 
                        'sanodep_0.0': 'SANODEP-$\lambda=0.0$', 
                        'sanodep_0.5': 'SANODEP-$\lambda=0.5$', 'gp':'GP', 
                        'sanodep_without_init_aug_0.0': 'SANODEP-$\lambda=0.0$ ',
                        'sanodep_without_init_aug_0.5': 'SANODEP-$\\backslash x_0$-$\\lambda=0.5$', 
                        'gp': 'GP'}
    tr_fc_pb = {'np_0.0': 0.0, 'np_0.5': 0.5, 'nodep_0.0': 0.0, 'sanodep_0.0': 0.0, 'sanodep_0.5': 0.5, 'sanodep_without_init_aug_0.0': 0.0, 'sanodep_without_init_aug_0.5': 0.5}
    yticks = {'mse': 
              {'LV': [0.2, 0.4, 0.6, 1.0], 
               'Brusselator': [0.0, 2.0, 4.0, 6.0, 8.0, 10], 
               'Selkov': [0.01, 0.05, 0.1], 
               'SIR': [0.0, 0.4, 0.6, 1.0], 
               'LV3D': [0.2, 0.4, 0.6, 1.0], 
               'SIRD': [0.2, 0.4, 0.6, 1.0]},} 
    mse_values = {}
    nll_values = {}

    fig_save_dir = f'exps/experiments/figs'
    try:
        os.makedirs(fig_save_dir)
    except:
        pass

    # Define a formatter to remove the exponent from y-axis and display clean values
    def format_func(value, tick_number):
        return f'{int(value)}'  # Display y-ticks as integers
    
    # Function to find the minimum order of magnitude for the y-values
    def find_common_exponent(y_values):
        min_value = np.min(np.abs(y_values[np.nonzero(y_values)]))  # Ignore zero values
        exponent = int(np.floor(np.log10(min_value)))
        return exponent
    
    for pb in pb_list:
        mse_values[pb] = {}
        nll_values[pb] = {}
        for model_names in model_list:
            mse_values[pb][model_names] = {}
            nll_values[pb][model_names] = {}
            if model[model_names] == 'nodep':
                workdir = f'exps/experiments/{problem_name_mapping[pb]}/{model[model_names]}/forcast_prob{tr_fc_pb[model_names]}' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/{model[model_names]}/forcast_prob{tr_fc_pb[model_names]}/'
            elif model[model_names] == 'gp':
                workdir = f'exps/experiments/{problem_name_mapping[pb]}/{model[model_names]}/seed_0' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/{model[model_names]}/seed_0/'
            else:
                workdir = f'exps/experiments/{problem_name_mapping[pb]}/{model[model_names]}/forcast_prob{tr_fc_pb[model_names]}' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/{model[model_names]}/forcast_prob{tr_fc_pb[model_names]}/'
            
            for eval_forecast_prob in ['0.0', '1.0']:
                mse_values[pb][model_names][eval_forecast_prob] = {}
                nll_values[pb][model_names][eval_forecast_prob] = {}


                for ctx_traj_num in range(0, test_traj_num):
                    mse_values[pb][model_names][eval_forecast_prob][ctx_traj_num] = []
                    nll_values[pb][model_names][eval_forecast_prob][ctx_traj_num] = []
                    seed_upper_lim = 1 if model[model_names] == 'gp' else 5
                    for seed in range(0, seed_upper_lim):
                        try:
                            # Read the CSV file
                            if model[model_names] == 'gp':
                                file_path = os.path.join(workdir, "evaluations", f"ctx_traj_size_({ctx_traj_num}, {ctx_traj_num})/")
                                mses, nlls = read_batch_gp_csv_file_from_path(file_path, sys_id_range=50)
                                mse_values[pb][model_names][eval_forecast_prob][ctx_traj_num].append(mses)
                                nll_values[pb][model_names][eval_forecast_prob][ctx_traj_num].append(nlls)                            
                            else:
                                file_path = os.path.join(workdir, f"seed_{seed}", "evaluations", f"ctx_traj_size_({ctx_traj_num}, {ctx_traj_num})/evaluation_metrics_forcst_prob_{eval_forecast_prob}.csv")  # Adjust the file name pattern to your needs
                                df = pd.read_csv(file_path)
                                df = df.dropna()

                                mse_values[pb][model_names][eval_forecast_prob][ctx_traj_num].append(df[df["MSE"] <= df["MSE"].quantile(stability_quantile)]["MSE"].mean())
                                nll_values[pb][model_names][eval_forecast_prob][ctx_traj_num].append(df[df["NLL"] <= df["NLL"].quantile(stability_quantile)]["NLL"].mean())
                        except:
                            pass

    def custom_log_formatter(val, pos):
        if val == 1:
            return r"$10^{0}$"
        elif val in [0.1, 10, 100, 0.01, 0.001, 1000]:  # Adjust based on the scales you are using
            return f"$10^{{{int(np.log10(val))}}}$"
        else:
            return f"${val:.1g} \\times 10^{{{int(np.log10(val))}}}$"
    
    # Plot the MSE w.r.t ctx_traj_num
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(pb_list),
        figsize=(12, 3.5),  # Increased height for better readability
        squeeze=False,
        # constrained_layout=True,  # Automatically adjust subplot params,
        # constrained_layout_kw={'w_pad': 0.1, 'h_pad': 0.5}  # Adjust padding
    )
    
    for row_idx, eval_forcast_prob in enumerate(['1.0', '0.0']):
        for pb_idx, pb in enumerate(pb_list):
            ax = axs[row_idx, pb_idx]  # Reference to the current subplot
    
            for model_names in model_list:  # Simplified loop
                mean_mse = []
                std_mse = []
    
                for ctx_traj_num in range(test_traj_num):
                    mse = mse_values[pb][model_names][eval_forcast_prob][ctx_traj_num]
                    if len(mse) != 0:
                        # Convert the list to a JAX array
                        mse_jax = np.array(mse)  # JAX array
    
                        # Compute mean and std using JAX's functions
                        mean = np.mean(mse_jax).item()  # Convert to Python float
                        std = np.std(mse_jax).item()    # Convert to Python float
    
                        # Append to lists
                        mean_mse.append(mean)
                        std_mse.append(std)
    
                # Convert mean and std lists to standard Python lists of floats
                mean_mse = np.array(mean_mse)  # JAX array
                std_mse = np.array(std_mse)    # JAX array
    
                # Plot the mean MSE
                ax.plot(
                    np.arange(test_traj_num),
                    mean_mse,
                    label=model_label_name.get(model_names, model_names),  # Safe dictionary access
                    linewidth=0.5,  # Increased linewidth for better visibility
                    color=color_list_map[model_names]
                )
    
                # Plot the std deviation as a shaded area around the mean
                ax.fill_between(
                    np.arange(test_traj_num),
                    mean_mse - std_mse,
                    mean_mse + std_mse,
                    alpha=0.3 if model[model_names] != 'gp' else 0.1,  # Adjust alpha for better visibility
                    color=color_list_map[model_names],
                    linewidth=0
                )
    
            # Set titles and labels
            ax.set_title(problem_title.get(pb, pb), fontsize=10)  # Safe dictionary access
            ax.set_xticks([0, 5, 10])  # X ticks consistent across all plots
            ax.set_yscale('log')  # Logarithmic Y-scale for better scaling
    
            # Configure Major Ticks
            major_locator = ticker.LogLocator(base=10.0, numticks=10)
            major_formatter = ticker.LogFormatterSciNotation(base=10.0, labelOnlyBase=True)
            ax.yaxis.set_major_locator(major_locator)
            ax.yaxis.set_major_formatter(major_formatter)
    
            # Configure Minor Ticks
            minor_locator = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
            ax.yaxis.set_minor_locator(minor_locator)
            minor_formatter = ticker.LogFormatterSciNotation(base=10.0, labelOnlyBase=False)
            ax.yaxis.set_minor_formatter(minor_formatter)
    
            # Retrieve current y-axis limits
            ymin, ymax = ax.get_ylim()
    
            # Get major tick values within the current y-limits
            major_tick_values = major_locator.tick_values(ymin, ymax)
    
            # Define a small epsilon to account for floating point precision
            epsilon = 1e-10
    
            # Check if any major tick is within the y-limits
            major_ticks_present = any(
                (ymin - epsilon) <= tick <= (ymax + epsilon) for tick in major_tick_values
            )
    
            if major_ticks_present:
                # Hide minor tick labels to prevent overlap
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            else:
                # Show minor tick labels as they act as primary ticks
                ax.yaxis.set_minor_formatter(minor_formatter)
    
            # Adjust tick parameters to differentiate major and minor ticks
            ax.tick_params(
                axis='y',
                which='major',
                labelsize=8,
                length=7,
                width=1.5
            )
            ax.tick_params(
                axis='y',
                which='minor',
                labelsize=6,
                length=4,
                width=1
            )
            ax.tick_params(
                axis='x',
                which='both',
                labelsize=8,
                length=5,
                width=1
            )
    
            # Add grid lines for better readability
            ax.grid(
                True,
                which='both',
                # linestyle='--',
                # linewidth=0.5,
                # alpha=0.7
            )
            # some manually tuning 
            # print(pb)
            # if pb == 'LV':
            #     ax.set_ylim(0.1, ax.get_ylim()[1])
            # if pb == 'LV3D':
            #     ax.set_ylim(0.1, ax.get_ylim()[1])  
    
    # Create a unified legend without duplicates
    handles, labels = axs[0, 0].get_legend_handles_labels()
    for ax in axs.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels
    
    # Remove duplicate labels while preserving order
    by_label = OrderedDict(zip(labels, handles))
    
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(model_list),
        fontsize=8,
        markerscale=2
    )
    
    # Add common Y and X labels for the figure
    fig.supylabel('MSE', ha='center', fontsize=10)
    fig.supxlabel('Number of Context Trajectories', ha='center', fontsize=10)
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.95, wspace=0.35, hspace=0.33)
    # Save the figure with high DPI and proper bounding
    plt.savefig(
        os.path.join(fig_save_dir, "seq_model_performance_mse.png"),
        dpi=1000,  # Higher DPI for better quality
        bbox_inches='tight'  # Ensure nothing gets clipped
    )

    # Now, plot the NLL following the same scheme
    # Plot the NLL w.r.t ctx_traj_num
    fig_nll, axs_nll = plt.subplots(
        nrows=2,
        ncols=len(pb_list),
        figsize=(12, 3.5),
        squeeze=False,
    )

    for row_idx, eval_forecast_prob in enumerate(['1.0', '0.0']):
        for pb_idx, pb in enumerate(pb_list):
            ax = axs_nll[row_idx, pb_idx]

            all_nll_values = []  # Collect all NLL values to determine the data range

            for model_names in model_list:
                mean_nll = []
                std_nll = []

                for ctx_traj_num in range(test_traj_num):
                    nll = nll_values[pb][model_names][eval_forecast_prob][ctx_traj_num]
                    if len(nll) != 0:
                        # Convert the list to a numpy array
                        nll_np = np.array(nll)

                        # Collect NLL values
                        all_nll_values.extend(nll_np.tolist())

                        # Compute mean and std using numpy functions
                        mean = np.mean(nll_np)
                        std = np.std(nll_np)

                        # Append to lists
                        mean_nll.append(mean)
                        std_nll.append(std)

                # Convert mean and std lists to numpy arrays
                mean_nll = np.array(mean_nll)
                std_nll = np.array(std_nll)

                # Plot the mean NLL
                ax.plot(
                    np.arange(len(mean_nll)),
                    mean_nll,
                    label=model_label_name.get(model_names, model_names),
                    linewidth=0.5,
                    color=color_list_map[model_names]
                )

                # Plot the std deviation as a shaded area around the mean
                ax.fill_between(
                    np.arange(len(mean_nll)),
                    mean_nll - std_nll,
                    mean_nll + std_nll,
                    alpha=0.3 if model[model_names] != 'gp' else 0.1,  # Adjust alpha for better visibility
                    color=color_list_map[model_names]
                )

            # Set titles and labels
            ax.set_title(problem_title.get(pb, pb), fontsize=10)
            ax.set_xticks([0, 5, 10])

            # Use 'symlog' scale for NLL because it can be negative
            ax.set_yscale('symlog', linthresh=1)

            # Adjust y-limits to include all data comfortably
            ymin, ymax = ax.get_ylim()
            def flatten(lst):
                flat_list = []
                for item in lst:
                    if isinstance(item, list):
                        flat_list.extend(flatten(item))
                    else:
                        flat_list.append(item)
                return flat_list
            all_nll_values = flatten(all_nll_values)
            data_ymin = min(all_nll_values) if all_nll_values else ymin
            data_ymax = max(all_nll_values) if all_nll_values else ymax

            # Expand y-limits slightly for better visualization
            ymin = min(ymin, data_ymin * 1.1)
            ymax = max(ymax, data_ymax * 1.1)
            ax.set_ylim(ymin, ymax)

            # Generate y-ticks dynamically based on data range
            max_ticks = 6  # Maximum total number of ticks per plot

            # Determine if zero is within the data range
            include_zero = ymin <= 0 <= ymax
            allocated_ticks = max_ticks - (1 if include_zero else 0)

            # Collect negative and positive NLL values separately
            negative_values = [x for x in all_nll_values if x < -1]
            positive_values = [x for x in all_nll_values if x > 1]

            # Generate negative ticks
            if negative_values:
                neg_abs_values = [abs(x) for x in negative_values]
                neg_abs_min = min(neg_abs_values)
                neg_abs_max = max(neg_abs_values)
                neg_exponent_min = max(1, int(np.floor(np.log10(neg_abs_min))))
                neg_exponent_max = int(np.ceil(np.log10(neg_abs_max)))
                negative_exponents = list(range(neg_exponent_min, neg_exponent_max + 1))
                neg_num_exponents = len(negative_exponents)
            else:
                negative_exponents = []
                neg_num_exponents = 0

            # Generate positive ticks
            if positive_values:
                pos_min = min(positive_values)
                pos_max = max(positive_values)
                pos_exponent_min = max(1, int(np.floor(np.log10(pos_min))))
                pos_exponent_max = int(np.ceil(np.log10(pos_max)))
                positive_exponents = list(range(pos_exponent_min, pos_exponent_max + 1))
                pos_num_exponents = len(positive_exponents)
            else:
                positive_exponents = []
                pos_num_exponents = 0

            # Allocate ticks per side
            if neg_num_exponents > 0 and pos_num_exponents > 0:
                # Both negative and positive exponents present
                neg_ticks_allowed = allocated_ticks // 2
                pos_ticks_allowed = allocated_ticks - neg_ticks_allowed
            elif neg_num_exponents > 0:
                # Only negative exponents
                neg_ticks_allowed = allocated_ticks
                pos_ticks_allowed = 0
            elif pos_num_exponents > 0:
                # Only positive exponents
                neg_ticks_allowed = 0
                pos_ticks_allowed = allocated_ticks
            else:
                neg_ticks_allowed = 0
                pos_ticks_allowed = 0

            # Limit exponents to allocated ticks
            def limit_exponents(exponents, ticks_allowed):
                num_exponents = len(exponents)
                if num_exponents <= ticks_allowed:
                    return exponents
                else:
                    step = int(np.ceil(num_exponents / ticks_allowed))
                    limited_exponents = exponents[::step][:ticks_allowed]
                    return limited_exponents

            negative_exponents = limit_exponents(negative_exponents, neg_ticks_allowed)
            positive_exponents = limit_exponents(positive_exponents, pos_ticks_allowed)

            # Generate ticks
            negative_ticks = [-10 ** i for i in negative_exponents]
            positive_ticks = [10 ** i for i in positive_exponents]

            # Combine ticks
            all_ticks = negative_ticks + ([0] if include_zero else []) + positive_ticks
            # Remove ticks outside of ymin and ymax
            all_ticks = [t for t in all_ticks if ymin <= t <= ymax]
            # Remove duplicates and sort
            all_ticks = sorted(set(all_ticks))

            # Set the y-axis major locator to use only the specified ticks
            ax.yaxis.set_major_locator(ticker.FixedLocator(all_ticks))
            ax.yaxis.set_minor_locator(ticker.NullLocator())  # Remove minor ticks

            # Define a custom formatter function
            def symlog_tick_formatter(t, pos):
                if t == 0:
                    return '0'
                else:
                    sign = '-' if t < 0 else ''
                    t_abs = abs(t)
                    exponent = np.log10(t_abs)
                    exponent_int = int(round(exponent))
                    if exponent_int >= 1:
                        return f'{sign}$10^{{{exponent_int}}}$'
                    else:
                        # For exponents less than 1, do not label
                        return ''

            # Set the y-axis major formatter to use the custom formatter
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(symlog_tick_formatter))

            # Remove ticks with empty labels
            all_ticks_labels = [symlog_tick_formatter(t, None) for t in all_ticks]
            all_ticks_with_labels = [(t, label) for t, label in zip(all_ticks, all_ticks_labels) if label != '']
            if include_zero:
                all_ticks_with_labels.append((0, '0'))

            # Update ticks and labels
            if all_ticks_with_labels:
                ticks, labels = zip(*all_ticks_with_labels)
                ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
                ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
            else:
                ax.yaxis.set_major_locator(ticker.NullLocator())

            # Adjust tick parameters
            ax.tick_params(
                axis='y',
                which='major',
                labelsize=8,
                length=7,
                width=1.5
            )
            ax.tick_params(
                axis='x',
                which='both',
                labelsize=8,
                length=5,
                width=1
            )

            # Add grid lines for better readability
            ax.grid(
                True,
                which='both',
            )

    # Create a unified legend without duplicates
    # [Legend code remains unchanged]
    handles, labels = axs_nll[0, 0].get_legend_handles_labels()
    for ax in axs_nll.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels

    # Remove duplicate labels while preserving order
    by_label = OrderedDict(zip(labels, handles))

    fig_nll.legend(
        by_label.values(),
        by_label.keys(),
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(model_list),
        fontsize=8,
        markerscale=2
    )
    # Add common Y and X labels for the figure
    fig_nll.supylabel('NLL', ha='center', fontsize=10)
    fig_nll.supxlabel('Number of Context Trajectories', ha='center', fontsize=10)
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.95, wspace=0.35, hspace=0.33)

    # Save the figure with high DPI and proper bounding
    plt.savefig(
        os.path.join(fig_save_dir, "seq_model_performance_nll.png"),
        dpi=1000,
        bbox_inches='tight'
    )




if __name__ == '__main__':
    plot_interpolating_and_forecasting_for_models()