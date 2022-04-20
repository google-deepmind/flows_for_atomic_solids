# Flows for atomic solids

The code in this repository can be used to train normalizing flow models
to generate samples of atomic solids, as described in [1]. It also
contains a Colab notebook that loads parameters of already trained models
and samples from them, plotting observables similar to the figures in the
paper.

## Installation and usage

### Structure of the code

The code is organized in the following folders:

* `colab`: contains a Colab notebook to explore the samples from pre-trained models.
* `experiments`: configuration files for Lennard-Jones and monatomic water experiments, and the script to run training on them.
* `models`: modules to build normalizing flow models.
* `systems`: definitions of the Lennard-Jones and monatomic water potentials used to train the models.
* `utils`: utilities for building lattices and computing observables from the model samples.

### Training a model

Python version >= 3.7 is required to install and run the code.

To train one of the normalizing flows described in the paper,
first clone the deepmind-research repository in a folder of your choice:

```shell
git clone https://github.com/deepmind/flows_for_atomic_solids.git
```

Set up a Python virtual environment with the required dependencies by running
the `run.sh` script. This will also test-run the training script to
make sure the installation succeeded.

```shell
source ./flows_for_atomic_solids/run.sh
```

Then run the `experiments/train.py` script, selecting one of the
pre-configured systems:

```shell
python -m flows_for_atomic_solids.experiments.train --system='lj_32'
```

Please note that a GPU is necessary to train the larger models.

### Exploring a pre-trained model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/flows_for_atomic_solids/blob/master/colab/explore_trained_models.ipynb
)

The Colab notebook `colab/explore_trained_models.ipynb` can be used
to access parameters of a set of models that have been trained as described
in the paper. The Colab will load the model and reproduce the energy,
radial distribution and work figures, as well as compute a free-energy
estimate.

## References

[1] Peter Wirnsberger, George Papamakarios, Borja Ibarz, Sébastien Racanière, Andrew J. Ballard, Alexander Pritzel and Charles Blundell.
*Normalizing flows for atomic solids*. [arXiv:2111.08696](
https://doi.org/10.48550/arXiv.2111.08696)


## Disclaimer

This is not an official Google product.
