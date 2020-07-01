# Bayesian Allocation Model: The Experiments - Part 1

The experiments in this part include:

- The synthetic data experiments presented in the first subsection of the results
- The international relations data experiments presented in the second subsection of the results

## Software requirements:

The experiments in this section are conducted using Linux OS and Julia programming language version 1.0.5, as well as Jupyter Notebook/Lab with the Julia kernel. Please visit https://julialang.org/ for information regarding installing Julia and its Jupyter notebook kernel.

After installing Julia 1.0.5, the packages required by the experiments can be installed by executing the script requirements.jl running the command in this folder:

julia requirements.jl

Note: If the installation of packages fails you might try again by appending the argument " latest" without quotes to the end of the command. If the packages install without a problem you do not need to do this.

### Synthetic experiments

The synthetic experiments can be replicated by running all cells in the file Experiments-Synthetic.ipynb. Results produced by this document are written into the folder results/synt/. After this notebook's run, another notebook, Visualizations-Synthetic.ipynb, can then be used to produce visualizations based on the results recorded. These visualizations can be viewed inside the notebook as well as saved inside the folder img/synt/.

## Experiments with international relations data

As described in the article, the international relations data is prepared in five different tensors according to the quantile cut off points for counting events between nations. The data for these five tensors can be found in data/icews/ folder as .csv files. Each file is also accompanied by two dictionaries for converting integers to country names or action types, if needed.
