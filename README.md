# learninghospital
Combining Deep RL and SimPy models of hospital systems

[![DOI](https://zenodo.org/badge/277261877.svg)](https://zenodo.org/badge/latestdoi/277261877)

## Run on BinderHub:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MichaelAllen1966/learninghospital/master)


## Set up environment locally

To get the correct libraries and versions it is recommended that the provided conda environment is used. To create and activate the `rlsimpy` environment used:

To create environment. Navigate to the `binder` folder.

`conda env create -f environment.yml`

To activate environment:

`conda activate rlsimpy`

To deactivate:

`conda deactivate`

To update environment (from updated yml file):

`conda env update --prefix ./env --file environment.yml  --prune`

To remove the environemnt:

`conda env remove -n rlsimpy`
