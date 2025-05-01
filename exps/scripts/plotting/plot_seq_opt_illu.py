import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.contour import QuadContourSet
from matplotlib.collections import PolyCollection, PathCollection
from functools import partial
from jax import random
import jax
import importlib
from jax import numpy as np

# Assuming lotka_voterra_1d_observer is correctly imported
from NeuralProcesses.objectives import lotka_voterra_1d_observer


def plot_the_obj_contour(axs, vmin, vmax):
    """
    Plot the real objective contour on the provided Axes.

    Parameters:
        axs (matplotlib.axes.Axes): The target Axes to plot on.
        vmin (float): Minimum value for contour normalization.
        vmax (float): Maximum value for contour normalization.
    """

    # Load config
    def load_config_from_py(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()  # Call the get_config function

    np_config_file = os.path.join('exps', 'cfgs', 'lotka_voterra_1d_visualization', 'np.py')
    config = load_config_from_py(np_config_file)

    init_cond_range = (config.experimental_design.x0_lower_bound[0], config.experimental_design.x0_upper_bound[0])
    time_range = config.data.args.t_range

    ratio = np.linspace(init_cond_range[0], init_cond_range[1], 100)[..., None]  # [state_dim]
    time = np.linspace(time_range[0], time_range[1], 1000)
    test_problem_rng = random.PRNGKey(0)

    observer = partial(
        lotka_voterra_1d_observer, 
        problem_rng=test_problem_rng,
        time_scaling=config.data.args.time_scaling_coefficient
    )

    def observer_with_first_state(times, init_cond, t0, t1):
        state = observer(init_cond, np.atleast_1d(times), t0, t1)
        return np.squeeze(state)

    # Get ground truth trajectory
    traj = jax.vmap(
        jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
        in_axes=(None, 0, None, None),
    )(time, ratio, time_range[0], time_range[1])

    # Create a meshgrid for contour plotting
    combined = np.hstack(
        (ratio.reshape(-1, 1), traj)
    )  # [initial_cond_num, num_timesteps + 1]

    # Sort combined by the ratio
    combined_sorted = combined[combined[:, 0].argsort()]
    traj_sorted = combined_sorted[:, 1:]

    X, Y = np.meshgrid(time, combined_sorted[:, 0])

    # Plot the ground truth contour
    contour2 = axs.contourf(X, Y, traj_sorted, levels=20, vmin=vmin, vmax=vmax, cmap='viridis')

    axs.set_yticks([0.5, 1.0, 1.5, 2.0])
    axs.set_title("Objective Function")
    axs.set_xlabel("Time")
    axs.set_ylabel("Initial Condition")
    plt.colorbar(contour2, ax=axs, orientation='vertical', ticks=[0, 2, 4, 6])


def extract_and_replot(loaded_fig, target_ax, scatter_size=50, tick_label_size=12):
    """
    Extracts plot elements from a loaded figure and replots them on the target Axes.

    Parameters:
        loaded_fig (matplotlib.figure.Figure): The loaded Matplotlib figure.
        target_ax (matplotlib.axes.Axes): The target subplot axes where the plot will be replotted.
        scatter_size (int or float, optional): Size of the scatter plot markers. Defaults to 50.
        tick_label_size (int or float, optional): Font size for tick labels. Defaults to 12.
    """
    for loaded_ax in loaded_fig.axes:
        print(f"\nProcessing loaded_ax: {loaded_ax}")

        # Synchronize axes limits
        xlim = loaded_ax.get_xlim()
        ylim = loaded_ax.get_ylim()
        target_ax.set_xlim(xlim)
        target_ax.set_ylim(ylim)
        print(f"Set target_ax limits to x: {xlim}, y: {ylim}")

        # Initialize a flag to check if contourf was added
        contourf_added = False

        # Iterate through all collections in the loaded axes
        for collection in loaded_ax.collections:
            print(f"  Checking collection of type: {type(collection)}")

            # Handle QuadContourSet
            if isinstance(collection, QuadContourSet):
                print("  Found QuadContourSet")
                # Iterate through the contour levels and their segments
                for level, segs in zip(collection.levels, collection.allsegs):
                    print(f"    Processing contour level: {level}")
                    # Determine facecolor based on colormap and normalization
                    norm = collection.norm
                    cmap = collection.cmap
                    facecolor = cmap(norm(level))
                    
                    for i, seg in enumerate(segs):
                        if len(seg) == 0:
                            continue  # Skip empty segments
                        # Create a Polygon patch
                        polygon = patches.Polygon(seg, facecolor=facecolor, edgecolor='none', zorder=1)
                        target_ax.add_patch(polygon)
                        contourf_added = True
                        print(f"      Added Polygon {i+1} with {len(seg)} vertices at level {level}")
        
            # Handle PolyCollection directly (if any)
            elif isinstance(collection, PolyCollection):
                print("  Found PolyCollection directly")
                polygons = [path.vertices for path in collection.get_paths()]
                colors = collection.get_facecolors()
                for i, (poly_vertices, color) in enumerate(zip(polygons, colors)):
                    polygon = patches.Polygon(poly_vertices, facecolor=color, edgecolor='none', zorder=1)
                    target_ax.add_patch(polygon)
                    contourf_added = True
                    print(f"    Added PolyCollection {i+1} with {len(poly_vertices)} vertices.")
        
            else:
                print(f"  Skipping collection of type: {type(collection)}")

        if not contourf_added:
            print("  No QuadContourSet or PolyCollection (contourf) found in loaded_ax.")

        # Copy lines
        for line in loaded_ax.get_lines():
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                zorder=3  # Ensure lines are above contourf
            )
            print(f"  Added line: label='{line.get_label()}', color={line.get_color()}")

        # Copy scatter plots
        scatter_added = False
        for collection in loaded_ax.collections:
            if isinstance(collection, PathCollection):
                print("  Found PathCollection (scatter plot)")
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    # Extract facecolors and edgecolors
                    facecolors = collection.get_facecolors()
                    edgecolors = collection.get_edgecolors()
                    # Determine the marker type
                    marker = 'o'  # Default marker
                    paths = collection.get_paths()
                    if paths:
                        marker_vertices = paths[0].vertices.tolist()
                        if len(marker_vertices) > 1:
                            marker = tuple(marker_vertices)
                    # Plot scatter with specified size
                    scatter = target_ax.scatter(
                        offsets[:, 0],
                        offsets[:, 1],
                        label=collection.get_label(),
                        color=facecolors[0] if facecolors.size > 0 else None,
                        marker=marker,
                        edgecolor=edgecolors[0] if edgecolors.size > 0 else None,
                        s=scatter_size,  # Apply scatter size here
                        zorder=2  # Ensure scatter is above contourf but below lines
                    )
                    scatter_added = True
                    print(f"    Added scatter plot: label='{collection.get_label()}', color={facecolors[0] if facecolors.size > 0 else None}, size={scatter_size}")

        if not scatter_added:
            print("  No scatter plots found in loaded_ax.")

        # Copy titles and labels
        title = loaded_ax.get_title()
        # Remove X and Y labels by setting them to empty strings
        target_ax.set_title("")  # Remove Title
        target_ax.set_xlabel("")  # Remove X label
        target_ax.set_ylabel("")  # Remove Y label
        # rmv ticks
        target_ax.set_xticks([])
        target_ax.set_yticks([])
        print(f"  Set target_ax title: '{title}', xlabel removed, ylabel removed")

        # Adjust tick label sizes
        target_ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        print(f"  Set tick label size to {tick_label_size}")

        # Copy legends if they exist
        if loaded_ax.get_legend() is not None:
            target_ax.legend()
            print("  Added legend to target_ax.")
    
        # After replotting, close the loaded figure to free memory
        plt.close(loaded_fig)

def plot_optimization_contour():
    """
    Create a multi-panel contour plot with:
    - Left: Ground truth contour.
    - Right: Model-based contours (GP, NP, SANODEP) across multiple trajectories.
    """
    traj_nums = 5  # Number of trajectories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    fig_dir = os.path.join(base_dir, 'experiments', 'figs')
    
    # Define directories for each model
    gp_dir = os.path.join(fig_dir, 'gp_bo')
    np_dir = os.path.join(fig_dir, 'np_bo')
    sanodep_dir = os.path.join(fig_dir, 'sanodep_bo')  # Replace with actual directory if different
    
    # Initialize the multi-panel figure with GridSpec
    fig = plt.figure(figsize=(12, 3))  # Keep figsize unchanged
    gs = GridSpec(
        nrows=3, 
        ncols=traj_nums + 2,  # +1 for the ground truth contour
        width_ratios=[1.5] + [0.1] + [1]*traj_nums,  # Ground truth is wider
        height_ratios=[1]*3, 
        figure=fig,
        left=0.05,   # Reduce left margin
        right=0.95,  # Reduce right margin
        top=0.85,    # Leave space at the top for row titles
        bottom=0.0, # Leave space at the bottom for x-label
        wspace=0.05, # Minimal horizontal space between subplots
        hspace=0.05   # Vertical space between subplots
    )
    
    # Left big plot (ground truth contour)
    ax_big = fig.add_subplot(gs[:, 0])
    plot_the_obj_contour(ax_big, vmin=0.0, vmax=7.0)  # Ensure vmin and vmax are set appropriately
    # ax_big.set_title("Ground Truth Contour", fontsize=12)
    
    # Define model information
    model_dirs = [gp_dir, np_dir, sanodep_dir]
    model_names = ['gp', 'NP', 'SANODEP']
    model_name_map = {'gp': 'GP', 'NP': 'NP', 'SANODEP': 'SANODEP'}
    
    # Iterate over each model (row)
    for row in range(3):  # Rows 0, 1, 2 for GP, NP, SANODEP respectively
        model_dir = model_dirs[row]
        model_name = model_names[row]
        
        # Iterate over each trajectory
        for traj_iter in range(traj_nums):
            ax = fig.add_subplot(gs[row, traj_iter + 1 + 1])  # +1 to skip the first column for big plot
            model_pickle_path = os.path.join(
                model_dir, 
                f'Lotka_Volterra_{model_name}_pred_across_traj_opt_step_{traj_iter + 1}.pkl'
            )
            if os.path.exists(model_pickle_path):
                try:
                    with open(model_pickle_path, 'rb') as f:
                        model_fig = pickle.load(f)
                    extract_and_replot(model_fig, ax, scatter_size=6, tick_label_size=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading {model_name} Fig\n{e}', 
                            ha='center', va='center', fontsize=8)
                    ax.axis('off')
                    print(f"Error loading {model_name} Fig for Trajectory {traj_iter + 1}: {e}")
            else:
                ax.text(0.5, 0.5, f'No {model_name} Fig', 
                        ha='center', va='center', fontsize=8)
                ax.axis('off')
                print(f"No {model_name} Fig found for Trajectory {traj_iter + 1}.")
    
    # Add Row Titles using fig.text for better positioning
    # row_titles = ['GP', 'NP', 'SANODEP']
    # for row, title in enumerate(row_titles):
    #     # Calculate the y-position for each row title
    #     # y=0 is bottom, y=1 is top. Adjust based on the number of rows and spacing.
    #     y_pos = 0.85 - (row + 0.5) * (0.85 - 0.15) / 3
    #     fig.text(0.5, y_pos, title, ha='center', va='center', fontsize=10, fontweight='bold')
    fig.text(0.26, 0.71, 'GP', ha='center', va='center', fontsize=10, rotation=90)
    fig.text(0.26, 0.44, 'NP', ha='center', va='center', fontsize=10, rotation=90)
    fig.text(0.26, 0.15, 'SANODEP', ha='center', va='center', fontsize=10, rotation=90)
    # Add 'Number of Trajectories' label at the bottom center
    fig.text(0.63, -0.1, 'Number of Evaluated Trajectories', ha='center', va='center', fontsize=9)
    
    # Adjust layout to prevent overlap
    # Using constrained_layout can also be considered, but here we fine-tune manually
    # plt.tight_layout()  # Removed to use manual GridSpec adjustments
    
    # Save the multi-panel plot
    output_path = os.path.join(fig_dir, 'seq_opt_contour.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory



# Example usage:
# plot_optimization_contour()



def test_save_and_load_matplotlib_with_pickle():
    os.chdir(os.path.dirname(__file__))
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    pickle_path = 'test_fig.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)
    with open(pickle_path, 'rb') as f:
        loaded_fig = pickle.load(f)
    # rmv pickle file
    os.remove(pickle_path)

if __name__ == "__main__":
    # test_save_and_load_matplotlib_with_pickle()
    plot_optimization_contour()
    