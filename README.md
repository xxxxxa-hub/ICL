# LLM_calib3

LLM_calib3 is a modular framework designed for running in-context learning (ICL) experiments with language models. The project includes modules for data handling, calibration of in-context examples, and inference, allowing researchers to conduct experiments across various datasets with different configurations.

## Project Structure

- **LLM_calib3/**  
  The root directory containing the main scripts, configuration files, and example notebooks.

- **ICL_modules/**  
  Contains essential utilities and helper functions:
  - data_loader.py: Handles data loading and preprocessing.
  - dataset_interface.py: Provides an interface for interacting with different datasets.
  - experiment_basics.py: Defines basic routines and experiment scaffolding.
  - functions.py and s_random.py: Contains various helper functions for experiments.
  
- **datasets/**  
  Holds dataset files for experiments (e.g., agnews, sst5, trec, rotten_tomatoes, financial_phrasebank). Users can add or update datasets as needed.

- **ICL_calibrations/**  
  Implements calibration methods for in-context learning:
  - calibration_methods.py: Defines different calibration strategies.
  - new_calib.py: Additional calibration routines.

- **ICL_inference/**  
  Contains modules for running inference with calibrated models:
  - inference.py: Executes the model inference and logs results.

- **Other Files**  
  - my_param_config.json: A sample parameter configuration file.
  - Example1.ipynb and Example2.ipynb: Example notebooks demonstrating how to use the framework.
  - run_experiments.py: Main script to run experiments with various command-line parameters.

## Installation

1. **Clone the Repository:**
   git clone https://github.com/yourusername/LLM_calib3.git
   cd LLM_calib3

2. **Set Up Your Environment (optional but recommended):**
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
   pip install -r requirements.txt

## Running Experiments

The main script to run experiments is run_experiments.py. You can specify various parameters via command-line arguments. Here’s an example command:

python run_experiments.py --num_seeds 1 --datasets agnews --k_values 4 --test_samples 512 --param_config my_param_config.json --methods Baseline --models Llama

### Command-Line Arguments Explanation

- **--num_seeds**:  
  The number of random seeds for running experiments, ensuring reproducibility.

- **--datasets**:  
  Specifies the dataset(s) to use. For example, agnews selects the AG News dataset.

- **--k_values**:  
  Sets the value of k for experiments (e.g., the number of examples per prompt).

- **--test_samples**:  
  The number of test samples to evaluate the model on (e.g., 512 samples).

- **--param_config**:  
  The path to a JSON configuration file containing additional parameters (e.g., my_param_config.json).

- **--methods**:  
  Chooses the calibration or experimental method to apply (e.g., Baseline). Additional methods may be available in the calibration modules.

- **--models**:  
  The model to be used for inference. For example, Llama indicates the use of the Llama language model.

Note: The arguments are flexible and can be adjusted based on your experimental needs. Modify them as necessary to suit your dataset, model, and calibration method.

## Examples

Check out Example1.ipynb and Example2.ipynb for example workflows and usage demonstrations.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you encounter bugs or have feature suggestions.

## License

[Include your license information here.]

## Contact

For any questions or feedback, please contact [Your Contact Information].
