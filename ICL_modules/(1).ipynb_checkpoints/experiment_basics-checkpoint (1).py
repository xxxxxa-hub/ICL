from . import  functions, s_random, dataset_interface
import copy
import functools

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
        self.demonstration_sampler = [random.sample_index_set(self._k, len(self.dataset.demonstration))]*len(self.dataset.test)
        
        self._default_repeat_times = repeat_times
        self._repeat_times = repeat_times
        self.metrics = metrics
        self.label_dis = [0] * len(self.dataset.get_label_space())
        self.predictions = []
     
    

        
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
    
    def get_repeat_times(self):
        return self._repeat_times

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

        # INFERENCE
        if preentered_prediction is None and forward_inference is not None:
            print("\nStart testing the forward inference function " + str(forward_inference) + " on the dataset: " + str(self.dataset.dataset_name))
            if not batched_inference:
                for index in range(len(self.dataset.test)):
                    prompt = self.get_prompts_for_test_sample(index)
                    result = forward_inference(prompt = prompt, label_space = self.dataset.test.get_label_space()) 
                    ground_truth.append(self.dataset.get_ground_truth_label_index(index,test_set = True))
                    self.label_dis[ground_truth[-1]] += 1
                    prediction.append(result)
                    print("\r", end="")
                    print("Process: {}%, {} in {}".format(
                        int((index ) / total_samples * 100), 
                        (index ), 
                        total_samples
                    ), ">>" * int(index) / (total_samples * 32), end="")
            else:
                prompts = []
                for index in range(len(self.dataset.test)):
                    prompts.append(self.get_prompts_for_test_sample(index))
                    ground_truth.append(self.dataset.get_ground_truth_label_index(index, test_set = True))
                    self.label_dis[ground_truth[-1]] += 1
                prediction = forward_inference(prompt = prompts, label_space = self.dataset.test.get_label_space())
                
        elif preentered_prediction is not None:
            for index in range(len(self.triplet_dataset.test)):
                ground_truth.append(self.dataset.get_default_ground_truth_label_index(index))
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