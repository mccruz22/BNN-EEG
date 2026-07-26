# BNN EEG

Python-based project for Biological Neural Network modeling and analysis for EEG/MNIST dataset. 

---

## Installation

Copy the project code and data (if using EEG data). Deidentified data that supports our finding are available from the authors upon reasonable request. Code completely runs with MNIST dataset if you don't have an EEG dataset.

## Environtment-Setup Using Conda

```
conda env create -f environment.yml

conda activate bnn-stages

python -m ipykernel install --user --name=bnn-stages --display-name "Python (bnn)"
```

## Environtment-Setup Using Pip

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

Misc: ipykernel

## Files in BNN-EEG Folder (Codes Folder)

- run_model.ipynb - training; run using Jupyter or can convert to.py
- Analyze_BNN_Anesthesia.ipynb - accuracies, weights, etc.
- BNN_*.pth - training files (when trained)
- HebbianLearning.ipynb - running and visualizing gradients

## Files in /bnbp5 Folder

##### For Classification
  - bnn_intralayer.py: all models (same for all codes)
  - mnist_spike_train_sliding*.py: input adjustments (two versions)
  - trainnospikemne*.py: dataset and training (same train functions, different load files for different datasets)
    - self.load_anest() or self.load_mnist() depending on dataset
    - change load_anest() depending on dataset


## Files in /data Folder

  - pretrained gradients to run HebbianLearning.ipynb

## License

The custom analysis code supporting the findings of this study is openly available in `https://github.com/mccruz22/BNN-EEG`, under MIT License Copyright (c) 2026 Madelyn Cruz license.
