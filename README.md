# LLM_calib3
**LLM_calib3** is a modular framework designed for running in-context learning (ICL) experiments with large language models (LLMs). It provides an end-to-end pipeline—from data handling and preprocessing to calibration and inference—allowing researchers and practitioners to conduct robust experiments across various datasets and configurations.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Command-Line Arguments](#command-line-arguments)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
LLM_calib3 is built to streamline the process of experimenting with in-context learning by providing:
- **Data Handling:** Easy-to-use modules for loading and preprocessing various datasets.
- **Calibration Methods:** A suite of calibration techniques to adjust model predictions and improve reliability.
- **Inference Modules:** Efficient routines for running inference with calibrated models.
- **Modularity:** Clear separation of concerns with dedicated modules for each major component (data, calibration, inference), allowing for easy extensions and modifications.

## Project Structure

```bash
LLM_calib3/
├── ICL_modules/
│   ├── data_loader.py         # Handles data loading and preprocessing.
│   ├── dataset_interface.py   # Provides an interface for interacting with different datasets.
│   ├── experiment_basics.py   # Contains experiment scaffolding and basic routines.
│   ├── functions.py           # Helper functions for experiments.
│   └── s_random.py            # Utilities for randomized operations.
├── datasets/
│   ├── agnews/                # Example dataset directories (agnews, sst5, trec, etc.).
│   └── ...                    # Add or update datasets as needed.
├── ICL_calibrations/
│   ├── calibration_methods.py # Implements various calibration strategies.
│   └── new_calib.py           # Additional calibration routines.
├── ICL_inference/
│   └── inference.py           # Executes model inference and logs results.
├── my_param_config.json       # Sample parameter configuration file.
├── Example1.ipynb             # Notebook demonstrating usage and workflows.
├── Example2.ipynb             # Additional example notebook.
└── run_experiments.py         # Main script to run experiments via command-line.
```bash


## Installation
1. **Clone the Repository:**
   `git clone https://github.com/yourusername/LLM_calib3.git`
   `cd LLM_calib3`

2. **Set Up Your Environment (Optional but Recommended):**
   Create and activate a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate`  (On Windows: `venv\Scripts\activate`)

3. **Install Dependencies:**
   `pip install -r requirements.txt`

## Usage
### Running Experiments
The main script to run experiments is `run_experiments.py`. This script allows you to control key parameters of your experiment using command-line arguments. For example, to run an experiment on the AG News dataset with a single seed, 4-shot ICL, and using the Llama model with a specific calibration method, you can run:
`python run_experiments.py --num_seeds 1 --datasets agnews --k_values 4 --test_samples 512 --param_config my_param_config.json --methods Baseline --models Llama`

### Command-Line Arguments
- **`--num_seeds`**: The number of random seeds to use in the experiment (ensures reproducibility). Example: `--num_seeds 5`
- **`--datasets`**: Specifies the dataset(s) to be used. If not provided, all available datasets are processed. Example: `--datasets agnews sst2`
- **`--k_values`**: Sets the value(s) of *k*, representing the number of in-context learning examples. Example: `--k_values 4 8`
- **`--test_samples`**: Number of test samples to evaluate on (default 512). Example: `--test_samples 600`
- **`--param_config`**: Path to a JSON file with additional experiment parameters. Example: `--param_config my_param_config.json`
- **`--methods`**: Calibration/experimental methods to apply (Baseline, CC, Domain, Batch, LR, etc.). Example: `--methods Baseline CC`
- **`--models`**: Which model(s) to use for inference. Defaults to Qwen, Llama, and Mistral. Example: `--models Llama Mistral`

## Examples
For a quick start, see the notebooks:
- **Example1.ipynb**: Demonstrates a basic workflow with default settings.
- **Example2.ipynb**: Shows advanced usage with multiple datasets and calibration methods.


## License


## Contact
For any questions or feedback, please contact:
**Korel Gundem**
Email: [korelgundem@gwu.edu]
