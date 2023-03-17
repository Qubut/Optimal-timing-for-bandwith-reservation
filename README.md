# Optimal timing prediction using LSTM and Transformer models

This script trains LSTM and Transformer models to predict future values of a time series. 

The LSTM model is implemented in the `models/lstm.py` file

and the Transformer model is implemented in the `models/transformer.py` file. 

The data is processed using the `utils/data_processor.py` file

and batches are generated using the `utils/batch_generator.py` file.

The training is done using the `trainers/trainer.py` file.
Dependencies

The script's dependencies are in `pyproject.toml` file

## Installation

To install the required dependencies, run the following command:

```sh

poetry install
```

## Usage

    The train.csv and test.csv files should be placed in the same directory as the script.
    To train the LSTM and Transformer models, run the following command:

```sh

poetry run python paper.py
```
    To create visualizations for the experiment results, run the following command:

```sh

poetry run python plot.py
```


## Output

    The paper.py file outputs the training loss of the LSTM and Transformer models.
    The plot.py file creates multiple visualizations using the visualisations/plotting.py file to visualize the experiment results.