universe = vanilla
executable = /homes/jqing/codes/NeuralProcesses/exps/scripts/model_comparison/model_comparison_eval_gp_condor.sh
Requirements = regexp("^(arc|beech|curve|edge|gpu|oak|point|ray|texel|vertex|willow)[0-9]{2}", TARGET.Machine)
output = fc0.0_id0.out
error = fc0.0_id0.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 0
queue
output = fc0.0_id1.out
error = fc0.0_id1.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 1
queue
output = fc0.0_id2.out
error = fc0.0_id2.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 2
queue
output = fc0.0_id3.out
error = fc0.0_id3.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 3
queue
output = fc0.0_id4.out
error = fc0.0_id4.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 4
queue
output = fc0.0_id5.out
error = fc0.0_id5.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 5
queue
output = fc0.0_id6.out
error = fc0.0_id6.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 6
queue
output = fc0.0_id7.out
error = fc0.0_id7.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 7
queue
output = fc0.0_id8.out
error = fc0.0_id8.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 8
queue
output = fc0.0_id9.out
error = fc0.0_id9.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 9
queue
output = fc0.0_id10.out
error = fc0.0_id10.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 10
queue
output = fc0.0_id11.out
error = fc0.0_id11.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 11
queue
output = fc0.0_id12.out
error = fc0.0_id12.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 12
queue
output = fc0.0_id13.out
error = fc0.0_id13.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 13
queue
output = fc0.0_id14.out
error = fc0.0_id14.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 14
queue
output = fc0.0_id15.out
error = fc0.0_id15.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 15
queue
output = fc0.0_id16.out
error = fc0.0_id16.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 16
queue
output = fc0.0_id17.out
error = fc0.0_id17.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 17
queue
output = fc0.0_id18.out
error = fc0.0_id18.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 18
queue
output = fc0.0_id19.out
error = fc0.0_id19.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 0.0 20 0 10 19
queue
output = fc1.0_id0.out
error = fc1.0_id0.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 0
queue
output = fc1.0_id1.out
error = fc1.0_id1.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 1
queue
output = fc1.0_id2.out
error = fc1.0_id2.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 2
queue
output = fc1.0_id3.out
error = fc1.0_id3.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 3
queue
output = fc1.0_id4.out
error = fc1.0_id4.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 4
queue
output = fc1.0_id5.out
error = fc1.0_id5.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 5
queue
output = fc1.0_id6.out
error = fc1.0_id6.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 6
queue
output = fc1.0_id7.out
error = fc1.0_id7.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 7
queue
output = fc1.0_id8.out
error = fc1.0_id8.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 8
queue
output = fc1.0_id9.out
error = fc1.0_id9.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 9
queue
output = fc1.0_id10.out
error = fc1.0_id10.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 10
queue
output = fc1.0_id11.out
error = fc1.0_id11.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 11
queue
output = fc1.0_id12.out
error = fc1.0_id12.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 12
queue
output = fc1.0_id13.out
error = fc1.0_id13.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 13
queue
output = fc1.0_id14.out
error = fc1.0_id14.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 14
queue
output = fc1.0_id15.out
error = fc1.0_id15.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 15
queue
output = fc1.0_id16.out
error = fc1.0_id16.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 16
queue
output = fc1.0_id17.out
error = fc1.0_id17.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 17
queue
output = fc1.0_id18.out
error = fc1.0_id18.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 18
queue
output = fc1.0_id19.out
error = fc1.0_id19.err
arguments = ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ 1.0 20 0 10 19
queue
