#!/usr/bin/env python
import os

# Suppress Hugging Face Hub progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import inspect
import json
import pickle
import random
import functools
import copy
import torch
import numpy as np

# Import your package modules
from ICL_modules import data_loader, dataset_interface, experiment_basics, functions
from ICL_inference import inference
from ICL_calibrations import calibration_methods

# Import transformers for model/tokenizer loading
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your main experiment runner function
from run import run_multiple_calibration_experiments_generic  # Adjust the import path as necessary


def load_datasets(selected_datasets=None, test_samples=512, split_seed=107):
    """
    Auto-detect and load datasets.
    If selected_datasets is provided (as a list), only those datasets are loaded.
    Otherwise, all datasets (that can be loaded) are used.
    """
    dataset_loaders = {
        name: func for name, func in inspect.getmembers(data_loader, inspect.isclass)
        if not name.startswith("_")
    }
    
    all_splitted_datasets = []
    
    for name, load_fn in dataset_loaders.items():
        # If user specified datasets, only load the ones that match
        if selected_datasets is not None and name not in selected_datasets:
            continue
        try:
            print(f"Loading and splitting: {name}")
            dataset = load_fn(from_cache=True)
            train_samples = len(dataset) - test_samples
            splitted = dataset_interface.DatasetSplitter(dataset, train_samples, test_samples, split_seed)
            all_splitted_datasets.append(splitted)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
    
    return all_splitted_datasets


def load_models(model_names):
    """
    Load models and tokenizers using the Hugging Face transformers library.
    For recognized model names, the corresponding MODEL_ID and HF_TOKEN are used.
    Default models are:
      - Qwen: "Qwen/Qwen2-7B-Instruct"
      - Llama: "meta-llama/Llama-2-7b-chat-hf"
      - Mistral: "mistralai/Mistral-7B-Instruct-v0.3"
    """
    models_to_run = {}
    
    for model_name in model_names:
        name_lower = model_name.lower()
        if name_lower in ["qwen", "qwen2", "qwen2-7b", "qwen2-7b-instruct"]:
            MODEL_ID = "Qwen/Qwen2-7B-Instruct"
            HF_TOKEN = "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"
        elif name_lower in ["llama", "llama2", "llama-2", "llama-2-7b", "llama-2-7b-chat-hf"]:
            MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
            HF_TOKEN = "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"
        elif name_lower in ["mistral", "mistralai", "mistralai-7b", "mistralai-7b-instruct-v0.3"]:
            MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN = "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"
        else:
            print(f"Model '{model_name}' not recognized. Skipping.")
            continue

        print(f"Loading {model_name} model... This may take a while if not cached.")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_auth_token=HF_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN
        ).eval()

        models_to_run[model_name] = (model, tokenizer)
    
    return models_to_run


def load_param_config(param_config_path, k_values):
    """
    Load parameter configuration from JSON if provided.
    If not, use a default config for k=4.
    """
    if param_config_path is None:
        # Default configuration for k=4 only.
        if 4 in k_values:
            print("Using default param_dic for k=4.")
            return {"4": {
                        "0": [90, 10],
                        "1": [90, 10],
                        "2": [90, 10],
                        "3": [90, 10]
                      }
                    }
        else:
            raise ValueError(
                "No param_config provided and default config is only available for k=4. "
                f"Please provide a JSON file with param_dic for k_values: {k_values}"
            )
    else:
        print(f"Loading param_dic from {param_config_path}")
        with open(param_config_path, "r") as f:
            param_dic = json.load(f)
        # Convert keys to integers if necessary
        param_dic = {int(k): {int(sub_k): v for sub_k, v in sub_dic.items()} for k, sub_dic in param_dic.items()}
        return param_dic


def main():
    parser = argparse.ArgumentParser(description="Submit calibration experiment jobs.")
    
    # Number of seeds to run (from predefined list)
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds to run (max 10 from the predefined list).")
    
    # List of datasets to run (if not provided, run on all)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="List of dataset names to run. If not provided, runs on all available datasets.")
    
    # k_values for number of demonstrations (default is 4)
    parser.add_argument("--k_values", type=int, nargs="+", default=[4],
                        help="List of k values for number of demonstrations (default is [4]).") 


    parser.add_argument("--lr_k_shot", type=int, default=6,
                        help="The maximum number of k-shots for a calibration learner.")
    parser.add_argument("--ablation", action="store_true",
                        help="Enable ablation experiments setting.")
    
    # Parameter configuration dictionary (JSON file) or use default config if not provided
    parser.add_argument("--param_config", type=str, default=None,
                        help="Path to a JSON file with the param_dic configuration.")
    
    # Calibration methods to run (default: all methods)
    parser.add_argument("--methods", nargs="*", default=["Baseline", "CC", "Domain", "Batch", "LR"],
                        help="List of calibration methods to run (default is all methods).")
    
    # Models to run (default: 3 models: Qwen, Llama, Mistral)
    parser.add_argument("--models", nargs="*", default=["Qwen", "Llama", "Mistral"],
                        help="List of model names to run experiments on.")
    
    parser.add_argument("--exp_name", type=str, default='',
                        help="Experiment name for tracking.")
    
    parser.add_argument("--test_samples", type=int, default=512,
                        help="Number of test samples to use when splitting datasets (default is 512).")
    parser.add_argument("--test_in_context_samples", type=int, default=24,
                        help="Maximum number of demonstration combination to use for inference.")
    
    args = parser.parse_args()
    print(args)
    
    # Predefined seeds list
    predefined_seeds = [
        2206632489, 2481609806, 24520513, 1825229417, 2411013676,
        241047738, 3736665164, 167757907, 1532401219, 1486393352
    ]
    
    if args.num_seeds > len(predefined_seeds):
        print(f"Requested {args.num_seeds} seeds but only {len(predefined_seeds)} available. Using maximum available.")
        args.num_seeds = len(predefined_seeds)
    seeds = predefined_seeds[:args.num_seeds]
    
    # Load datasets (using inspect-based auto-detection)
    all_splitted_datasets = load_datasets(
        selected_datasets=args.datasets,
        test_samples=args.test_samples,
        split_seed=107
    )
    if not all_splitted_datasets:
        print("No datasets were loaded. Exiting.")
        return
    
    # Load parameter configuration (either default or from file)
    param_dic = load_param_config(args.param_config, args.k_values)
    
    # Load models and tokenizers
    models_to_run = load_models(args.models)
    if not models_to_run:
        print("No valid models loaded. Exiting.")
        return
    
    # Dictionary to store results for each model
    final_results = {}
    
    # Run experiments for each model
    for model_name, (model, tokenizer) in models_to_run.items():
        print(f"\n================ Running experiments for model: {model_name} ================")
        results_dic, dfs, coefficients_dic = run_multiple_calibration_experiments_generic(
            model=model,
            tokenizer=tokenizer,
            splitted_datasets=all_splitted_datasets,
            seeds=seeds,
            k_values=args.k_values,
            param_dic=param_dic,
            methods_to_run=args.methods,
            lr_k_shot=args.lr_k_shot,
            ablation=args.ablation,
            test_in_context_samples=args.test_in_context_samples,
        )
        final_results[model_name] = {
            "results_dic": results_dic,
            "dfs": dfs,
            "coefficients_dic": coefficients_dic
        }
    
    # Print a summary of the results
    print("\nExperiments complete. Summary of results:")
    for model_name, res in final_results.items():
        dataset_keys = list(res["results_dic"].keys())
        print(f"Model: {model_name} -> Datasets processed: {dataset_keys}")
        
    # print(final_results)
    
    # Save final results to a pickle file
    if not os.path.exists("results"): # save to a folder named "results"
        os.makedirs("results")
    save_file_path = os.path.join("results", f"results_k_{args.k_values}_seeds_{args.num_seeds}_datasets_{args.datasets}_models_{args.models}_{args.exp_name}.pkl")
    
    with open(save_file_path, "wb") as f:
        pickle.dump(final_results, f)
    print("\n Results Saved")


if __name__ == "__main__":
    main()
