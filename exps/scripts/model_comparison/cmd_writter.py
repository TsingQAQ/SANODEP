'''
Code to write .cmd file
'''

with open("eval_gp.cmd", "w") as f:

    f.write('universe = vanilla\n')
    f.write('executable = /homes/jqing/codes/NeuralProcesses/exps/scripts/model_comparison/model_comparison_eval_gp_condor.sh\n')
    # f.write('environment = "GRB_LICENSE_FILE=/homes/yx2822/license.lic"\n')
    f.write(
        'Requirements = regexp("^(arc|beech|curve|edge|gpu|oak|point|ray|texel|vertex|willow)[0-9]{2}", TARGET.Machine)\n')

    for forecast_prob in [0.0, 1.0]:
        for process_idx in range(20):

            f.write(f'output = fc{forecast_prob}_id{process_idx}.out\n')
            f.write(f'error = fc{forecast_prob}_id{process_idx}.err\n')
            f.write(
                f'arguments = ./exps/cfgs/brusselator/gp.py exps/experiments/brusselator/gp/ {forecast_prob} 20 0 10 {process_idx}\n')
            f.write('queue\n')

    f.close()

