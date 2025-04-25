from ICL_modules import data_loader, dataset_interface, s_random, experiment_basics, functions
from ICL_inference import inference
from ICL_calibrations import calibration_methods
import random
import torch
import numpy as np
import pandas as pd
import itertools
import functools
import copy
import time

def run_multiple_calibration_experiments_generic(model,tokenizer,splitted_datasets, seeds, k_values, param_dic, methods_to_run=None):
    """
    Generic version of the calibration runner function with all supported calibration methods.

    Parameters:
      splitted_datasets: Either a single dataset object or a list of dataset objects.
      seeds (list of int): List of random seeds.
      k_values (list of int): List of values for number of demonstrations.
      param_dic (dict): Dataset-specific or generic hyperparameter config for LR calibration.
          Example structure (dataset-specific):
              {
                  'sst2': { 4: {0: [...], ...}, 8: {0: [...], ...} },
                  'agnews': { 4: {...}, 8: {...} }
              }
          Or a generic one:
              {
                  4: {...},
                  8: {...}
              }
      methods_to_run (list of str): Calibration methods to use. If None, all are used.

    Returns:
      Tuple of dictionaries: (results_dic, dfs, coefficients_dic)
    """

    if not isinstance(splitted_datasets, list):
        splitted_datasets = [splitted_datasets]

    if methods_to_run is None:
        methods_to_run = ["Baseline", "CC", "Domain", "Batch", "LR"]

    results_dic = {}
    dfs = {}
    coefficients_dic = {}

    for splitted_dataset in splitted_datasets:
        dataset_key = splitted_dataset.dataset_name
        print(f"Starting experiments on dataset: {dataset_key}")

        # Determine whether to use dataset-specific or shared param_dic
        if dataset_key in param_dic:
            dataset_param_dic = param_dic[dataset_key]
        else:
            print(f"No dataset-specific param_dic found for '{dataset_key}', using generic config.")
            dataset_param_dic = param_dic

        if list(dataset_param_dic.keys()) != k_values:
            raise ValueError(f"param_dic for {dataset_key} and k_values do not match.\nDefine parameter configuration for each case.")

        results_dic[dataset_key] = {}
        dfs[dataset_key] = {}
        coefficients_dic[dataset_key] = {}

        for seed in seeds:
            results_dic[dataset_key][f'seed_{seed}'] = {}
            dfs[dataset_key][f'seed_{seed}'] = {}
            coefficients_dic[dataset_key][f'seed_{seed}'] = {}

            for k in k_values:
                print(f"\n=== Running experiment for {dataset_key} | Seed {seed}, k: {k} ===")
                experiment = experiment_basics.Experiment(dataset=splitted_dataset, k=k, seed=seed)
                demonstration_set_index = experiment.demonstration_sampler[0]

                inference_base = functools.partial(
                    inference.standard_ICL_inference2,
                    model=model,
                    tokenizer=tokenizer,
                    cache_empty=torch.cuda.empty_cache,
                    return_hidden_state=False,
                    return_full_vocab_prob=False
                )

                results_dic[dataset_key][f'seed_{seed}'][f'{k}'] = {}

                if "Baseline" in methods_to_run:
                    print(f"Running Baseline...")
                    results_dic[dataset_key][f'seed_{seed}'][f'{k}']['Baseline'] = experiment.run_experiment(inference_base)

                if "CC" in methods_to_run:
                    print(f"Running Contextual Calibration (CC)...")
                    calib_cc = calibration_methods.contextual_calibration(experiment.get_label_space())
                    calib_cc.train(
                        experiment.prompt_writer,
                        inference_base,
                        experiment.demonstration_set(),
                        k=k,
                        demonstration_set_index=demonstration_set_index
                    )
                    inference_cc = functools.partial(
                        inference.standard_ICL_inference,
                        model=model,
                        tokenizer=tokenizer,
                        cache_empty=torch.cuda.empty_cache,
                        return_hidden_state=False,
                        return_full_vocab_prob=False,
                        calibration_function=calib_cc
                    )
                    results_dic[dataset_key][f'seed_{seed}'][f'{k}']['CC'] = experiment.run_experiment(inference_cc)

                if "Domain" in methods_to_run:
                    print(f"Running Domain Calibration...")
                    calib_domain = calibration_methods.domain_calibration(experiment.get_label_space())
                    calib_domain.train(
                        experiment.prompt_writer,
                        inference_base,
                        experiment.demonstration_set(),
                        k=k,
                        calibration_number=20,
                        demonstration_set_index=demonstration_set_index
                    )
                    inference_domain = functools.partial(
                        inference.standard_ICL_inference,
                        model=model,
                        tokenizer=tokenizer,
                        cache_empty=torch.cuda.empty_cache,
                        return_hidden_state=False,
                        return_full_vocab_prob=False,
                        calibration_function=calib_domain
                    )
                    results_dic[dataset_key][f'seed_{seed}'][f'{k}']['Domain'] = experiment.run_experiment(inference_domain)

                if "Batch" in methods_to_run:
                    print(f"Running Batch Calibration...")
                    inference_batch = functools.partial(
                        inference.batched_ICL_inference,
                        model=model,
                        tokenizer=tokenizer,
                        cache_empty=torch.cuda.empty_cache,
                        batch_calibration_function=calibration_methods.batch_calibration,
                        inside_calibration_function=None
                    )
                    results_dic[dataset_key][f'seed_{seed}'][f'{k}']['Batch'] = experiment.run(inference_batch, batched_inference=True)

                if "LR" in methods_to_run:
                    print("Running LR Calibration")
                    g_average_voting = {}
                    lr_k_shot = 6 if k in [8,16] else k

                    for i in range(lr_k_shot):
                        experiment = experiment_basics.Experiment(dataset=splitted_dataset, k=k, seed=seed)
                        demonstration_set_index = experiment.demonstration_sampler[0]
                        dem = copy.deepcopy(demonstration_set_index)

                        print(f"Training LR for {i}-shot with index: {demonstration_set_index}")

                        start = time.time()
                        tempcali = calibration_methods.lr_calib_scipy_1d_cos(  
                            experiment.get_label_space(),
                            use_invariance=True,
                            lambda_invariance=dataset_param_dic[k][i][1],
                            invariance_loss_type='sym_ce',
                            constraint=True,
                            k=i,
                            max_iter=1000,
                            dic=dataset_param_dic,
                            cosine_threshold=np.cos(np.pi/dataset_param_dic[k][i][0]).item(),
                            verbose=True
                        )

                        df = tempcali.train(
                            experiment.prompt_writer,
                            inference_base,
                            experiment.demonstration_set(),
                            k=i,
                            demonstration_set_index=demonstration_set_index
                        )
                        end = time.time()
                        elapsed_minutes = (end - start) / 60
                        print(f"Training Elapsed time: {elapsed_minutes:.2f} minutes")

                        df['true_label_likelihood'] = df.apply(
                            lambda row: row['score_1'] if row['label'] == 1 else row['score_0'], axis=1
                        )
                        dfs[dataset_key][f'seed_{seed}'][f'{i}'] = df
                        params = tempcali.params_
                        coefficients_dic[dataset_key][f'seed_{seed}'][f'{i}'] = tempcali._unpack_params(params)

                        lr_results = {
                            'accuracy': [],
                            'averaged_truelabel_likelihood': [],
                            'macro_F1': [],
                            'expected_calibration_error_1': [],
                            'roc_auc': []
                        }
                        prob_index = {}
                        random.seed(42)
                        np.random.seed(42)

                        print('Selecting demonstration set for LR...')
                        if i != 0:
                            my_dem = tempcali._permutate(dem, i-1)
                            sample_size = min(len(my_dem) // 2,12)
                            my_dem = random.sample(my_dem, sample_size)
                        else:
                            my_dem = [1]

                        start = time.time()
                        for dem_ind in my_dem:
                            if i == 0:
                                dem_ind = []
                            else:
                                dem_ind = dem_ind[:i]

                            experiment.set_demonstration_sampler([list(dem_ind)] * 512)
                            print(f"Testing LR, Seed {seed}, k: {i}, demo index: {experiment.demonstration_sampler[0]}")
                            my_inference_2 = functools.partial(
                                inference.standard_ICL_inference2,
                                model=model,
                                tokenizer=tokenizer,
                                cache_empty=torch.cuda.empty_cache,
                                return_hidden_state=False,
                                return_full_vocab_prob=False,
                                calibration_function=tempcali
                            )
                            result0, probs = experiment.run_experiment(my_inference_2, return_outputs=True)

                            if i == 0:
                                results_dic[dataset_key][f'seed_{seed}'][f'{k}'][f'LR-{i}'] = result0
                                g_average_voting[f'LR-{i}'] = probs['prob.']
                                break
                            else:
                                prob_index[tuple(dem_ind)] = probs['prob.']
                                results_dic[dataset_key][f'seed_{seed}'][f'{k}'][f'LR-{i}-{dem_ind}'] = result0
                                lr_results['accuracy'].append(result0['accuracy'])
                                lr_results['averaged_truelabel_likelihood'].append(result0['averaged_truelabel_likelihood'])
                                lr_results['macro_F1'].append(result0['macro_F1'])
                                lr_results['expected_calibration_error_1'].append(result0['expected_calibration_error_1'])
                        end = time.time()
                        elapsed_minutes = (end - start) / 60
                        print(f"Testing Elapsed time: {elapsed_minutes:.2f} minutes")

                        if i == 0:
                            continue

                        average_voting = functions.average_probabilities(prob_index)
                        g_average_voting[f'LR-{i}'] = average_voting.tolist()
                        print(f"Averaging probabilities for LR-{i}")
                        result_avg = experiment.run_experiment(input_prediction=average_voting.tolist())
                        results_dic[dataset_key][f'seed_{seed}'][f'{k}'][f'LR-{i}'] = {
                            'accuracy': np.mean(lr_results['accuracy']),
                            'averaged_truelabel_likelihood': np.mean(lr_results['averaged_truelabel_likelihood']),
                            'macro_F1': np.mean(lr_results['macro_F1']),
                            'expected_calibration_error_1': np.mean(lr_results['expected_calibration_error_1']),
                            'roc_auc': np.mean(lr_results['roc_auc']) if lr_results['roc_auc'] else None
                        }
                        results_dic[dataset_key][f'seed_{seed}'][f'{k}'][f'LR-{i}-average_voting'] = result_avg
                        print(result_avg['accuracy'])

                    print('Final general average voting for LR')
                    weights = functions.compute_weights_for_k_shot(k, first_k_shot=lr_k_shot)
                    final_avg = functions.average_probabilities(g_average_voting, weights=weights)
                    result_final_avg = experiment.run_experiment(input_prediction=final_avg.tolist())
                    results_dic[dataset_key][f'seed_{seed}'][f'{k}'][f'LR-average_voting'] = result_final_avg

    return results_dic, dfs, coefficients_dic
