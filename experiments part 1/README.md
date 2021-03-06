# Bayesian Allocation Model: The Experiments - Part 1

The experiments in this part include:

- The synthetic data experiments presented in the first subsection of the experiments section of the paper.
- The international relations data experiments presented in the second subsection of the experiments section of the paper.

## Software requirements:

The experiments in this section are conducted using Linux OS and Julia programming language version 1.0.5, as well as Jupyter Notebook/Lab with the corresponding Julia kernel. Please visit https://julialang.org/ for information regarding installing Julia and its Jupyter notebook kernel.

After installing Julia 1.0.5, the packages required by the experiments can be installed by executing the script `requirements.jl`:

`julia requirements.jl`

Note: If the installation of packages fails, you might try again with the command (if the packages installed the first time without a problem you do not need to do this): 

`julia requirements.jl latest`

### Synthetic experiments

The synthetic experiments can be replicated by running all cells in the file `Experiments-Synthetic.ipynb`. Results produced by this document are written into the folder `results/synt/`. After this notebook's run, another notebook, `Visualizations-Synthetic.ipynb`, can then be used to produce visualizations based on the results recorded. These visualizations can be viewed inside the notebook as well as saved inside the folder `img/synt/`.

## Experiments with international relations data

As described in the article, the ICEWS international relations data is prepared in five different tensors according to the quantile cut off points for counting events between nations. The data for these five tensors can be found in `data/icews/` folder as `.csv` files. Each file is also accompanied by two dictionaries for converting integers to country names or action types, if needed.

The main interface for running ICEWS experiments is the file `icews_experiments.jl`. This file accepts parameters from both inside the file and from the command line. The parameters inside the file include:

- `data_name`: this is the name of the data file (without the `.csv` extension), see `data/icews/` folder for your options.
- `exp_type`: this is a custom name for the experiment currently conducted (you can freely name your experiment)
- `meths`: these are the methods that will be used in the experiments. any selection of methods, or all of them can be provided in this list.
- `M`: How much should each experiment be repeated for each method. For SMC methods an average of the results is taken, while for VB the maximum ELBO is taken. The length of this list must equal hat of `meths`, since it specifies the number of repetitions for each method.
- `adaptive`: whether the experiment should include adaptive resampling (as opposed to regular resampling).
- `resampling_freq`: if the resampling is not adaptive, this number denotes the period for the resampling. Set it to 1 for resampling at every step, e.g. 10 for once in each ten steps, set to total number of tokens for no resampling.

The rest of the experiment parameters are given when running the file. The file is run using the command:

`julia --depwarn=no <a> <N> <EPOCHS> <R_start> <R_end>`

`<a>` should be a float denoting the equivalent sample size hyperparameter (see the paper for details)
`<N>` should be an integer denoting the number of particles for SMC (must be entered as placeholder even if no SMC methods are used)
`<EPOCHS>` should be an integer denoting the number of EPOCHS for VB (must be entered as placeholder even if no VB methods are used)
`<R_start>` and `<R_end>` are integers that denote the model orders between for which the experiments will be conducted.

A numerical example of this command would be:

`julia --depwarn=no 1.0 10000 100 2 16`

This would correspond to an experiment where a=1.0, N=10000, EPOCHS=100 and Rs=2:16. Note: Each CP model has a single R; for a Tucker decomposition any R will be interpreted as (R, R, R) - see the supplementary material for an example of such experiment results.

After the experiment is conducted and completed, the script writes the numerical results along with experiment metadata in a `.json` file. The path of the file would be:

`results/<data_name>_<exp_name>/exp_a_<a>_N_<N>_EPOCHS_<EPOCHS>_f_<resamling_freq>_Rs_<R_start>:<R_end>.json`

You can read, examine, and visualize results using your preferred software tool. You can also use the script `icews_visualize.jl` to conduct this visualisation after the results are produced. `icews_visualize.jl` takes the result file path to a `json` produced by `icews_experiments.jl` as the argument, such that:

`julia icews_visualize.jl <path>/<to>/<your>/<results>/<file>.json`

This script produces an image in a similar folder structre path: `img/<data_name>_<exp_name>/exp_a_<a>_N_<N>_EPOCHS_<EPOCHS>_f_<resamling_freq>_Rs_<R_start>:<R_end>.pdf`

This can be used for repeating the experiments provided in the article, or conducting other experiments (e.g. experiments with different a's than reported by the paper or the supplementary material.)

### Exact replication of results

For the _exact_ numerical replications of the experiments, some additional considerations is required. Since many experiments were conducted for the paper, they were separated into different runs, recombination of which requires some additional work. We automatized the majority of this work for the ease of the reader. We provide the additional instructions for this exact replication in the file `Exact_Replication.md`.
