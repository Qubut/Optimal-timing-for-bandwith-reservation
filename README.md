# Optimal Timing Prediction for Bandwidth Reservation

this project is an experiment to predict the prices for bandwidth reservation from different Network operators at a given moment using deep learning techniques. By harnessing the power of LSTM and Transformer architectures, this solution achieves high accuracy in its predictions, catering to diverse datasets and time series complexities.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Features

- **LSTM Model**: Utilizes Long Short-Term Memory networks, a type of recurrent neural network, for handling long-term dependencies in time series data.
- **Transformer Model**: Employs the Transformer architecture, known for its efficiency in sequence transduction tasks.


## Project Structure


- datasets/: Contains the raw data files.
- models/: Contains the neural network architectures for both LSTM and Transformer models.
- config/: Configuration settings and hyperparameters.
- utils/: Utility functions for data processing and batching.
- trainers/: Training logic for the neural network models.
- visualisations/: Scripts and functions for visualizing training results.
- out/: Contains trained model and measurements for visualization


## Dependencies

All the dependencies required for this project are listed in the `pyproject.toml` file for those using Poetry, and in the `requirements.txt` file for pip users.

## Installation

### Using Poetry

1. If not installed, get [Poetry](https://python-poetry.org/).
2. From the project's root directory, install the dependencies:

```bash

poetry install

```
### Using pip

For those not using Poetry, dependencies can also be installed using pip:

```bash

pip install -r requirements.txt

```

## Usage

1. Model Training:
    Execute the paper.py script to train both the LSTM and Transformer models:
```sh
python paper.py

```

2. Result Visualization:

After training, you can visualize the results by executing the plot.py script:

```sh

python plot.py

```

you can also use poetry, eg: `poetry run python paper.py`

## Results

Once the models are trained, you'll find saved models and performance metrics in the out/ directory. Visualizations generated will provide insights into model performance, including loss and accuracy over time.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.