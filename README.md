# BNN EEG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22931816.svg)](https://doi.org/10.5281/zenodo.22931816)

Biologically Interpretable Machine Learning Approaches for Analyzing Neural Data

Python-based project for Biological Neural Network modeling and analysis for EEG/MNIST dataset. 

---

## Installation

Copy the project code and data (if using EEG data). Deidentified data that supports our finding are available from the authors upon reasonable request. Code completely runs with MNIST dataset if you don't have an EEG dataset.

## Environtment-Setup Using Conda

Run from the directory containing `environment.yml`:

```
conda env create -f environment.yml

conda activate bnn-stages

python -m ipykernel install --user --name=bnn-stages --display-name "Python (bnn)"
```

## Environtment-Setup Using Pip

Install Python 3.10 if it is not already available. Open Command Prompt
in the project directory and run:

```
py -3.10 -m venv bnn-stages
bnn-stages\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

python -m ipykernel install --user --name=bnn-stages --display-name "Python (bnn)"

```

## Dependencies

Numerical/Scientific: numpy, scipy, pandas, h5py, mne, matplotlib

Deep Learning: torch, torchvision, snntorch

Utilities: jaxtyping, zanj, muutils, albumentations, datasets

Misc: ipykernel, seaborn

## Files in BNN-EEG Folder (Codes Folder)

- run_model.ipynb - training; run using Jupyter or can convert to.py
- Analyze_BNN_Anesthesia.ipynb - accuracies, weights, etc.
- HebbianLearning.ipynb - running and visualizing gradients
- Representational geometry analysis.ipynb - plotting tSNE components using pretrained files (you may train your own models since pretrained models containing hidden layer activities are too large for GitHub or download the sample [file](https://drive.google.com/file/d/1CCvAxgLBJjQ6V_nLoj0aqqvAO0IKqYxu/view?usp=sharing) to data folder before running this script.)

## Model and training modules

The `bnbp5` directory contains:
- `bnn_intralayer.py`: neural network and neuron models
- `mnist_spiketrain_sliding.py`: dataset wrappers and spike-train preparation
- `trainnospikemne_intralayer_timeind_sliding_128.py`: dataset loading,
  training, evaluation, checkpoint handling, and gradient measurement

## Data directory

`data` or another configured dataset directory contains:

- MNIST files and generated spike caches
- Anesthesia recordings and the event spreadsheet
- pretrained gradients to run HebbianLearning.ipynb
- pretrained models to be able to run Analyze_BNN_Anesthesia.ipynb

## License

The custom analysis code supporting the findings of this study is openly available at `https://github.com/mccruz22/BNN-EEG`, under the GNU Affero General Public License v3.0 (AGPL-3.0).

This repository provides a reference/ illustrative implementation of the methods described in the paper Biologically Interpretable Machine Learning Approaches for Analyzing Neural Data. It is not the production system used in ongoing research.

