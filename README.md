# ICEtools

Deep/Machine Learning framework for **I**mputing **C**ircRNA **E**xpression value

## Prerequisites

## Installation

## Usage

### Predefined models

Predict the expression level of circRNA using a pre-defined deep learning model and the expression matrix of protein coding genes (PCGs) user provided.
```bash
ICE.py pre_defined [options] <new PCGs exp matrix>
```

### Custom models

Use the training data provided by the user to train the custom model. And then use the custom model and expression matrix of protein coding genes (PCGs) user provided to predict circRNA expression level.
```bash
ICE.py custom [options] <training PCGs mat> <training circRNA mat> <new PCGs exp matrix>
```

### Options

#### `--output_prefix <prefix>`

Define the output files prefix. Default is "output"

#### `--num_threads <int>`

Define the number of CPUs for parallel processing. Default is 4

#### `--new_circ <circ exp mat>`

Provide measured value matrix of circRNAs for calculating the correlation coefficient with the predicted value

#### `--disable_value_output`

Cancel output of predicted value. It is invalid when `--new_circ` does not exist

