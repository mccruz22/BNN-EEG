# BNN EEG

Python-based project for Biological Neural Network modeling and analysis for EEG/MNIST dataset. 

---

## Installation

Copy the project code and data (if using EEG data).


## Environtment-Setup

conda env create -f environment.yml

conda activate bnn-stages

python -m ipykernel install --user --name=bnn-stages --display-name "Python (bnn)"

## Dependencies

Numerical/Scientific: numpy, scipy, pandas, h5py, mne, matplotlib

Deep Learning: torch, torchvision, snntorch

Utilities: jaxtyping, zanj, muutils, albumentations, datasets

Misc: ipykernel

## Files in BNN-EEG Folder (Codes Folder)

- run_model.ipynb - training; run using Jupyter or can convert to.py
- Analyze_BNN_Anesthesia.ipynb - accuracies, weights, etc.
- BNN_*.pth - training files

## Files in /bnbp5 Folder

##### For Classification
  - bnn_intralayer.py: all models (same for all codes)
  - mnist_spike_train_sliding*.py: input adjustments (two versions)
  - trainnospikemne*.py: dataset and training (same train functions, different load files for different datasets)
    - self.load_anest() or self.load_mnist() depending on dataset
    - change load_anest() depending on dataset




