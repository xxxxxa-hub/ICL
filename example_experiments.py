#!/usr/bin/env python
# coding: utf-8

# # Main Experiment Runner for One Dataset Example
# 

# In[1]:


from ICL_modules import data_loader, dataset_interface, s_random, experiment_basics, functions
from ICL_inference import inference
from ICL_calibrations import calibration_methods
from run import run_multiple_calibration_experiments_generic
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    LogitsProcessor
)
import random
import torch
import numpy as np
import pandas as pd
import itertools
import functools
import copy
import accelerate
import pickle


# In[ ]:


def seed_everything(seed: int):
    """
    Sets the seed for all major sources of randomness to ensure reproducibility.
    
    Args:
        seed (int): The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"All sources of randomness have been seeded with: {seed}")

seed = 321
seed_everything(seed)


# In[2]:


model_name = "Qwen/Qwen2-7B-Instruct"
my_cache_dir = "/hpc/group/fanglab/xx102/hf-cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=my_cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", cache_dir=my_cache_dir).eval()


# In[5]:


hh = data_loader.hh_rlhf(tokenizer=tokenizer, from_cache = True)
# hh = data_loader.agnews(from_cache = True)
test_samples = 512
train_samples = len(hh) - test_samples
splitted_hh = dataset_interface.DatasetSplitter(hh, train_samples,test_samples,seed)


# In[ ]:


param_dic = {
#   4: {
#     0: [90, 10],
#     1: [90, 10],
#     2: [90, 10],
#     3: [90, 10]
#   },
  8: {
    0: [90, 2],
    1: [90, 2],
    2: [90, 2],
    3: [90, 2],
    4: [90, 2],
    5: [90, 2],
    6: [90, 2],
    7: [90, 2]
  },
  # 16: {
  #   0: [90, 10],
  #   1: [90, 10],
  #   2: [90, 10],
  #   3: [90, 10],
  #   4: [90, 10],
  #   5: [90, 10],
  #   6: [90, 10],
  #   7: [90, 10],
  #   8: [90, 10],
  #   9: [90, 10],
  #   10: [90, 10],
  #   11: [90, 10],
  #   12: [90, 10],
  #   13: [90, 10],
  #   14: [90, 10],
  #   15: [90, 10]
  # }
}


# In[9]:


# Run a calibration experiment using the Contextual Calibration (CC) method.
#
# Arguments:
# - model / tokenizer: HuggingFace-compatible model and tokenizer.
# - splitted_datasets: One or more dataset splits (train/test) ready for calibration.
# - seeds: List of seeds for random sampling of demonstration examples (for reproducibility).
# - k_values: List of values for k-shot learning (number of demonstrations in-context).
# - param_dic: Dictionary of hyperparameters (used mainly for LR).
# - methods_to_run: Specifies which calibration methods to run. Here, we use only 'LR'.
#
# Returns:
# - result: A dictionary containing performance metrics for each dataset, seed, and k.
# - The other two return values (ignored here) include:
#     - dfs: Calibration-specific dataframes (mostly used by LR).
#     - coefficients_dic: Calibration coefficients (also for LR).
#
# This call executes LR calibration on the provided model and dataset, printing progress logs as it runs.
result, _, _ = run_multiple_calibration_experiments_generic(
    model=model,
    tokenizer=tokenizer,
    splitted_datasets=splitted_hh,
    seeds=[seed],
    k_values=[8],
    param_dic=param_dic,
    methods_to_run=["Baseline", "CC", "Domain", "Batch", "LR"]
)


# In[10]:


print(result)
pickle.dump(result, open(f"result_seed_{seed}_k_8_lambda_2.pkl", "wb"))


# # Main Experiment Runner for All Datasets

# In[4]:


# import inspect

# # Settings
# test_samples = 512
# seed = 107

# # Auto-detect all dataset loader functions in data_loader (e.g., glue_sst2, agnews, yelp, etc.)
# dataset_loaders = {
#     name: func for name, func in inspect.getmembers(data_loader, inspect.isclass)
#     if not name.startswith("_")  # skip private functions
# }

# # Now split and store them
# all_splitted_datasets = []

# for name, load_fn in dataset_loaders.items():
#     try:
#         print(f"Loading and splitting: {name}")
#         dataset = load_fn(from_cache=True)
#         train_samples = len(dataset) - test_samples
#         splitted = dataset_interface.DatasetSplitter(dataset, train_samples, test_samples, seed)
#         all_splitted_datasets.append(splitted)
#     except Exception as e:
#         print(f"Skipping {name} due to error: {e}")


# In[22]:


# result, _, _ = run_multiple_calibration_experiments_generic(model=model,
#                                                             tokenizer=tokenizer,
#                                                             splitted_datasets=all_splitted_datasets,
#                                                             seeds=seeds[:2],
#                                                             k_values=[4],
#                                                             param_dic=param_dic,
#                                                             methods_to_run=['CC','Baseline']
#                                                         )


# In[23]:


# result

