from . import  functions, s_random, dataset_interface
import copy
import functools
import random
from typing import List

class Experiment():
    def __init__(self, 
        dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "accuracy": functions.accuracy,
            "averaged_truelabel_likelihood": functions.averaged_truelabel_likelihood,
            "macro_F1": functions.macro_F1,
            "expected_calibration_error_1": functions.expected_calibration_error_1
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = 1,
        seed : int = 107):
        if k < 0:
            raise ValueError("k should be a positive integer.")
        if repeat_times < 0:
            raise ValueError("repeat_times should be a positive integer.")
        
        
        self._k = k
        
        self.dataset = dataset
        self.seed = seed
        random = s_random.Random(seed)
        
        
        self.prompt_writer = dataset.prompt_writter
        self.demonstration_sampler = [self.random_sample_index(dataset.demonstration.get_dataset(),k,seed)]*len(dataset.test)
        
        self._default_repeat_times = repeat_times
        self._repeat_times = repeat_times
        self.metrics = metrics
        self.label_dis = [0] * len(self.dataset.get_label_space())
        self.predictions = []
        
    def random_sample_index(self, data, k, seed=None, class_proportions=None):
    
        if seed is not None:
            random.seed(seed)
        
        # Group indices by label.
        groups = {}
        for idx, (_, label) in enumerate(data):
            groups.setdefault(label, []).append(idx)
        
        labels = list(groups.keys())
        m = len(labels)
        
        # If no proportions are provided, use uniform distribution.
        if class_proportions is None:
            class_proportions = {label: 1.0 / m for label in labels}
        else:
            # Normalize provided proportions for the labels present in the dataset.
            total_prop = sum(class_proportions.get(label, 0) for label in labels)
            if total_prop == 0:
                class_proportions = {label: 1.0 / m for label in labels}
            else:
                class_proportions = {label: class_proportions.get(label, 0) / total_prop for label in labels}
        
        # Allocate initial picks for each label based on the desired proportions.
        picks = {}
        for label in labels:
            available = len(groups[label])
            # Calculate target count using floor division.
            target = int(k * class_proportions[label])
            picks[label] = min(target, available)
        
        # Distribute any remaining picks among groups that still have available examples.
        total_allocated = sum(picks.values())
        remaining = k - total_allocated
        
        label_order = labels.copy()
        random.shuffle(label_order)
        while remaining > 0:
            made_pick = False
            for label in label_order:
                if picks[label] < len(groups[label]):
                    picks[label] += 1
                    remaining -= 1
                    made_pick = True
                    if remaining == 0:
                        break
            if not made_pick:
                break  # No more picks can be allocated.
        
        # Randomly sample the allocated number of indices from each group.
        result_indices = []
        for label in labels:
            if picks[label] > 0:
                selected = random.sample(groups[label], picks[label])
                result_indices.extend(selected)
        random.shuffle(result_indices)
        return result_indices
            
        
    
     
    

        
    def run_experiment(self, forward_inference: callable = None, input_prediction = None, batched_inference = False, return_outputs = False):
        return self.run(forward_inference, preentered_prediction = input_prediction, batched_inference = batched_inference, return_outputs = return_outputs)
        
    
    

    def get_prompts_for_test_sample(self, test_sample_index: int):
        
        test_index = test_sample_index
        demos_indexes = self.demonstration_sampler[test_index]
        

        demonstrations = [self.dataset.demonstration.__getitem__(i) for i in demos_indexes] 
        query = self.dataset.test.get_input_text(test_index)
        
        return self.prompt_writer(demonstrations, query)

    def add_metric(self, metric_name: str, metric_function: callable):
        if metric_name in self.metrics:
            warnings.warn("The metric name already exists. Overwriting the metric function.")
        self.metrics[metric_name] = metric_function

    def get_k(self):
        return self._k
    
 

    def get_label_space(self):
        return copy.deepcopy(self.dataset.get_label_space())
    
    def set_demonstration_sampler(self, sampler: List[List[int]]):
    
        self.demonstration_sampler = sampler
        
    


    def run(
        self, 
        forward_inference: callable = None, 
            # When batched_inference is disabled, forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. The inputted parameter signs are fixed to prompt and label_space.
            # When batched_inference is enabled, forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
        preentered_prediction = None, 
            # The prediction for the test set (list[int]). If None, the prediction will be calculated by the forward_inference.
        batched_inference = False, 
            # for batched inference like BatchCalibration.
            # If enabled, we will input all the prompts into the forward_inference; and if disabled, we will input the prompt into the forward_inference one by one
        return_outputs = False 
            # If True, the outputs will be returned.    
    ):
       
       
        
      
        ground_truth = []
        prediction = []
        total_samples = len(self.dataset.test)
        bar_length = 32
        # INFERENCE
        if preentered_prediction is None and forward_inference is not None:
            print("\nStart testing the forward inference function on the dataset: " + str(self.dataset.dataset_name))
            if not batched_inference:
                for index in range(len(self.dataset.test)):
                    prompt = self.get_prompts_for_test_sample(index)
                    result = forward_inference(prompt = prompt, label_space = self.dataset.test.get_label_space()) 
                    ground_truth.append(self.dataset.get_ground_truth_label_index(index,test_set = True))
                    self.label_dis[ground_truth[-1]] += 1
                    prediction.append(result)
                    
                    # Calculate progress percentage (floating-point)
                    percentage = (index + 1) / total_samples * 100  # float percentage
                    
                    # Determine number of progress units
                    progress_units = max(1, total_samples // bar_length)
                    progress_bar = ">>" * ((index + 1) // progress_units)
                    
                    # # Print progress bar dynamically
                    # print(
                    #     "\rProcess: {:>5.1f}% | {:>5} / {} | {}".format(
                    #         percentage, index + 1, total_samples, progress_bar
                    #     ),
                    #     end=""
                    # )

            else:
                prompts = []
                for index in range(len(self.dataset.test)):
                    prompts.append(self.get_prompts_for_test_sample(index))
                    ground_truth.append(self.dataset.get_ground_truth_label_index(index, test_set = True))
                    self.label_dis[ground_truth[-1]] += 1
                prediction = forward_inference(prompt = prompts, label_space = self.dataset.test.get_label_space())
                
        elif preentered_prediction is not None:
            for index in range(len(self.dataset.test)):
                ground_truth.append(self.dataset.get_ground_truth_label_index(index))
                self.label_dis[ground_truth[-1]] += 1
            prediction = preentered_prediction
        else:
            raise ValueError("You should provide either the forward_inference function or the input_prediction.")

        # TEST
        self.predictions = prediction
        final_result = {}
        for metric_name, metric_function in self.metrics.items():    
            final_result[metric_name] = metric_function(ground_truth, self.predictions)

        if return_outputs:
            return final_result,  {"groundtruths": ground_truth, "predicted": functions.compress_logits_prediction_to_onehot(prediction), "prob.": self.predictions}
        return final_result
    

    def demonstration_set(self):
        return self.dataset.demonstration
    
    def test_set(self):
        return self.dataset.test