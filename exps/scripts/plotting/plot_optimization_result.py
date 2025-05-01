import pandas as pd
import glob
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)  # Defines flag for `ConfigDict` files compatible with absl flags

# Parse command line arguments
flags.DEFINE_string('config', default='', help='Maximum time per trajectory')
flags.DEFINE_string('opt_type', default='mo', help='Optimization type')


stability_epsilon = 0.02
color_list_map = {'gp': 'tab:blue', 'np_0.0': 'tab:orange', 'np_0.5': 'tab:brown', 'nodep': 'tab:green', 'sanodep_0.0': 'tab:pink', 'sanodep_0.5': 'tab:purple', 'grey_box_LV_sanodep': 'tab:red'}


def plot_interpolating_and_forecasting_for_models(argv):
    pb_list = ['LV', 'Brusselator', 'Selkov', 'SIR', 'LV3D', 'sird']
    problem_name_mapping = {
        'LV': 'lotka_voterra',
        'Brusselator': 'brusselator',
        'sird': 'sird',
        'Selkov': 'selkov',
        'SIR': 'sir_unnormalized',
        'LV3D': 'lotka_voterra_3d'
    }
    problem_title = {
        'LV': 'Lotka-Voterra ($2d$)',
        'Brusselator': 'Brusselator ($2d$)',
        'sird': 'SIRD ($4d$)',
        'Selkov': 'Selkov ($2d$)',
        'SIR': 'SIR ($3d$)',
        'LV3D': 'Lotka-Voterra ($3d$)'
    }
    legend_title = {
        'gp': r'GP',
        'sanodep_0.5': r'SANODEP-$\lambda = 0.5$',
        'grey_box_LV_sanodep': r'PI-SANODEP-$\lambda = 0.5$',
        'np_0.0': r'NP-$\lambda=0.0$',
        'np_0.5': r'NP-$\lambda=0.5$',
        'nodep': r'NODEP',
        'sanodep_0.0': r'SANODEP-$\lambda = 0$'
    }
    import importlib.util

    def load_config_from_py(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()  # Call the get_config function

    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 2))
    fig.text(0.005, 0.5, 'Hypervolume Indicator', va='center', rotation='vertical', fontsize=7)

    for pb_idx, pb in enumerate(pb_list):
        print(f'working on plot of problem: {problem_name_mapping[pb]}')
        for opt_model in ['np_0.0', 'np_0.5', 'nodep', 'sanodep_0.0', 'sanodep_0.5', 'gp']:
            if opt_model == 'gp':
                model = 'gp'
            elif opt_model == 'np_0.0' or opt_model == 'np_0.5':
                model = 'np'
            elif opt_model == 'sanodep_0.5' or opt_model == 'sanodep_0.0':
                model = 'sanodep'
            elif opt_model == 'nodep':
                model = 'nodep'
            elif opt_model == 'grey_box_LV_sanodep':
                model = 'grey_box_LV_sanodep'
            try:
                config_file = os.path.join('exps/cfgs/', problem_name_mapping[pb], f'{model}.py')
                config = load_config_from_py(config_file)
                workdir = f'exps/experiments/{problem_name_mapping[pb]}/' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/'
                plot_regret(config, opt_model, workdir, axs.flat[pb_idx], pb_idx, problem_title[pb], legend_title[opt_model])
            except:
                print(f'Failed to plot {opt_model}...')
    # Collect legend handles and labels
    lines_labels = [ax.get_legend_handles_labels() for ax in axs.flat]
    lines, labels = zip(*lines_labels)
    unique_dict = {}

    for lines, labels in zip(lines, labels):
        for _line, _label in zip(lines, labels):
            if _label not in unique_dict:
                unique_dict[_label] = _line

    unique_labels, unique_lines = zip(*unique_dict.items())

    # Place legend at the bottom center
    fig.legend(unique_lines, unique_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=7, markerscale=2)

    # Adjust subplot parameters
    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3, top=0.9, wspace=0.3, hspace=0.5)

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=6)

    # Save the figure
    current_script_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    plt.savefig(
        os.path.join(current_script_dir, 'experiments/figs', f"optimization_result.png"),
        dpi=2000,  # Increase dpi for higher resolution
    )


def plot_regret(config, opt_model, specified_workdir, axs, pb_idx, problem_title, legend_label):
    try:
        if opt_model == 'gp':
            workdir = os.path.join(specified_workdir, opt_model, 'opt', 'mo')
        elif opt_model == 'np_0.0':
            workdir = os.path.join(specified_workdir, 'np', 'forcast_prob0.0', 'seed_0', 'opt', 'meta_learn', 'mo')
        elif opt_model == 'np_0.5':
            workdir = os.path.join(specified_workdir, 'np', 'forcast_prob0.5', 'seed_0', 'opt', 'meta_learn', 'mo')
        elif opt_model == 'sanodep_0.5':
            workdir = os.path.join(specified_workdir, 'sanodep', 'forcast_prob0.5', 'seed_0', 'opt', 'meta_learn', 'mo')
        elif opt_model == 'sanodep_0.0':
            workdir = os.path.join(specified_workdir, 'sanodep', 'forcast_prob0.0', 'seed_0', 'opt', 'meta_learn', 'mo')
        elif opt_model == 'nodep':
            workdir = os.path.join(specified_workdir, opt_model, 'forcast_prob0.0', 'seed_0', 'opt', 'meta_learn', 'mo')
        elif opt_model == 'grey_box_LV_sanodep':
            workdir = os.path.join(specified_workdir, 'grey_box_LV_sanodep', 'forcast_prob0.5', 'seed_0', 'opt', 'meta_learn', 'mo')

        csv_files = glob.glob(os.path.join(workdir, 'optimization_log_*.csv'))

        xs = []
        ys = []

        # Read the first few lines of the file to get the reference_global_maximum
        first_lines = pd.read_csv(csv_files[0], nrows=5)

        # The reference_global_maximum is the value in the second column of the row where the first column is 'reference_global_maximum'
        reference_global_maximum = float(first_lines[first_lines['doe_init_conds'] == 'reference_global_maximum'].iloc[0, 1])

        for file in csv_files:
            # Read csv file
            df = pd.read_csv(file, skiprows=8)  # skip the first 7 rows
            x = []  # cumulative time
            y = []  # regret
            # Calculate cumulative time with shift and append to x
            total_prev_traj_eval_time = 0
            traj_count = 0
            for queried_time, regret in zip(df['queried_time'], df['regret']):
                if queried_time == 0:  # new trajectory starts
                    if traj_count == 0:  # do nothing on the first trajectory
                        pass
                    else:  # update past evaluated trajectoris time
                        total_prev_traj_eval_time += config.model.t1 * config.data.args.time_scaling_coefficient
                    traj_count += 1
                if queried_time == config.model.t1:
                    queried_time -= stability_epsilon
                x.append(total_prev_traj_eval_time + queried_time * config.data.args.time_scaling_coefficient)

                # Append regret values to y
                y.append(reference_global_maximum - regret)  # stack current best
            xs.append(x)
            ys.append(y)

        # Interpolate the regret through a uniform time grid
        uniform_time = np.linspace(config.model.t0, (config.model.t1 - config.model.t0) * config.data.args.time_scaling_coefficient * config.experimental_design.num_traj_iter * 0.99, 200)
        # Interpolate the regret on the uniform time grid
        interpolated_regret = []
        for _xs, _ys in zip(xs, ys):
            f = interp1d(_xs, _ys, kind='linear', fill_value="extrapolate")
            uniform_regret = f(uniform_time)
            interpolated_regret.append(uniform_regret)

        uniform_time = uniform_time / ((config.model.t1 - config.model.t0) * config.data.args.time_scaling_coefficient * config.experimental_design.num_traj_iter)
        # Plot the mean regret and fill between +-1 std
        axs.plot(uniform_time, np.mean(interpolated_regret, axis=0), label=f'{legend_label}', linewidth=0.3, color=color_list_map[opt_model])
        axs.fill_between(uniform_time, np.mean(interpolated_regret, axis=0) - np.std(interpolated_regret, axis=0), np.mean(interpolated_regret, axis=0) + np.std(interpolated_regret, axis=0), alpha=0.2, color=color_list_map[opt_model], edgecolor=None)
        axs.set_xticks(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        axs.tick_params(axis='both', which='major', labelsize=1)
    except:
        print(f'Failed to plot {opt_model}...')
    if pb_idx >= 0:
        axs.set_xlabel('Scaled Time', fontsize=7)
    axs.set_title(problem_title, fontsize=10)
    axs.grid(True)  # Enable grid on this subplot
    if problem_title == 'Brusselator ($2d$)':
        axs.set_ylim(0.5, axs.get_ylim()[1])
    if problem_title == 'Selkov ($2d$)':
        axs.set_ylim(2.0, axs.get_ylim()[1])


if __name__ == "__main__":
    app.run(plot_interpolating_and_forecasting_for_models)
