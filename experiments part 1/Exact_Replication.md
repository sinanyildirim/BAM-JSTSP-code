# Exact Numerical Replication of ICEWS Experiments

This file is a continuation of the README.md file for the Part 1 of experiments. This file focuses on the _exact_ numerical replications of the experiments (with random number generation controlled through random seeds).

Since many experiments were conducted for the paper, they were separated into different runs. For exact numerical replications, the replicating experiments should also follow the same separations (since otherwise the different seed settings would produce different, albeit very similar results). The original result files provided in the folder `results/`, ending with `*_original` will provide the blueprint for this. The three steps required for the replication of experiments are 1- conducting the experiments in the same separate runs, 2- consolidating results, 3- plotting the results (optional).

1- All result files that have been used in the paper can be found in the folder `results/` with endings `*_original`. Looking at the folder `results/icews_0.95_data_cp_original/` as an example, which include the comparison of SMC vs VB under the CP model at X_0.95, we can see that for experiments with a = 1.0, two separate files exist:

- `exp_a_1.0_N_17730_EPOCHS_100_f_0_Rs_2:17.json`
- `exp_a_1.0_N_17730_EPOCHS_100_f_0_Rs_18:25.json`

implying that the original experiments were completed in two runs, with Rs=2:17 and Rs=18:25 conducted separately. The replication attempt must also conduct the experiments in two different runs using same Rs, as well as the same number of particles (which is N = 17730, as can be read from file name).

So, for the experiments comparing the SMC vs VB using the X_0.95 data at a = 1.0 under the CP model, we need to first set the parameters inside the `icews_experiment.jl` file correctly. In order to make replication easier, we kept the original experiment files for the CP and TD experiments under the names `icews_experiments_original_cp.jl` and `icews_experiments_original_td.jl`, which are the same with the `icews_experiments.jl` file, but the experiment parameters are set to the settings in the paper. So to replicate the above experiments the user must run

- `julia --depwarn=no icews_experiments_original_cp.jl 1.0 17730 100 2 17`
- `julia --depwarn=no icews_experiments_original_cp.jl 1.0 17730 100 18 25`

Other results can be replicated in a similar manner by examining the original result folders to see how the experiments were broken up into different runs. (In order to avoid overwriting the original results, the two files mentioned above are set to write the results under folder names ending with `*_cp` and `*_td` instead of `*_cp_original` and `*_td_original`).

2- Now that we have the results, but in separate files, we have to consolidate them into a single file so that they are plottable and are easier to examine in general. You can use the file called `icews_consolidate.jl` for this task. Continuing on the example above, if we wanted to consolidate the experiments with a = 1.0 and N = 17730 above, in the folder `results/icews_0.95_data_cp_original/`, the correct syntax for the script would be:

`julia icews_consolidate.jl icews_0.95_data_cp_original 1.0 17740`

This creates a new file under the folder `results/consolidated/icews_0.95_data_cp_original/` called `exp_a_1.0_N_17730_EPOCHS_100_f_0_Rs_2:25.json`. See the consolidated versions of the original experiments under the `results/consolidated/` folder for examples. In a single experiment set with same a's and N's and different R's, R's must follow each other without gap or overlap for this consolidation to work. For example, the two part experiments above share the same a and N, and the R's perfectly follow each other with one being 2:17, the other 18:25. To provide further illustration, we provided the commands for the consolidation of all original experiments in the main paper and the supplement in a bash script called `consolidate_originals.sh`. You can edit and use the script for similar operations.

For any other original experiment replication beyond the example above, one must similarly use the original result folders to learn the number of particles N used, as well as how the R's were divided between different runs, and conduct their own experiments similarly. Our observation is that our results are quite robust to different random initializations, however we provide a path to complete replication for the sake of completeness.

3- The resulting consolidated files can be plotted by using the `icews_visualize.jl` file as describe in the main README.md file. The plots for the original results can be found in the `img/consolidated/` folder (created with the script `plot_all_originals.sh`).
