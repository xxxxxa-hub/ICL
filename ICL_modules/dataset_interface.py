from . import s_random
from . import data_loader
import copy
from typing import List


class DatasetSplitter:
    def __init__(self, dataset: data_loader.datasets_loader, demonstration_number, test_number, random_seed):
        self.dataset = dataset       
        self.random_seed = random_seed
        self.demonstration_number = demonstration_number
        self.test_number = test_number
        self.dataset_name = dataset.dataset_name

        random = s_random.Random(random_seed)
        ind = random.sample_index_set(demonstration_number + test_number, len(dataset))
        self.demonstration, self.test = dataset.split([ind[:demonstration_number], ind[demonstration_number:]])
        
    def demonstration_set(self):
        return self.demonstration
        
    def test_set(self):
        return self.test

    def get_label_space(self):
        # Return a deep copy of the label space.
        return copy.deepcopy(self.dataset.get_label_space())

    def get_ground_truth_label(self, index, test_set = True) -> str:
        if index < 0 or index >= len(self.test):
            raise ValueError("Index out of range.")
        if test_set:
            return self.test.get_label(index)
        else:
            return self.demonstration.get_label(index)
        
    def get_ground_truth_label_index(self, index, test_set = True) -> int:
        if index < 0 or index >= len(self.test):
            raise ValueError("Index out of range.")
        if test_set:
            return self.test.find_index_from_label(self.get_ground_truth_label(index, test_set=True))
        else:
            return self.demonstration.find_index_from_label(self.get_ground_truth_label(index, test_set = False))
            

    def prompt_writter(self, demonstrations: List[tuple] , query_line :List[str]):
        """
        Builds a prompt by iterating over demonstrations and then appending
        the query line. Incorporates all dataset-specific prefixes/affixes.
        """
        # Start with any instruction you might have
        prompt = self.dataset._instruction
    
        # Process each demonstration
        for text_list, label in demonstrations:
            # Get the index of this label in your label space
            # Get the index of this label in your label space
            label_index = self.dataset.find_index_from_label(label)
            # Convert that index into the actual label text
            label_str = self.dataset._label_space[label_index]
            
    
            # Append text + label according to your configured prefixes/affixes
            prompt += (
                f"{self.dataset._input_text_prefixes[0]}"
                f"{text_list[0]}"  # If text_list is always a single element, or do ' '.join(text_list)
                f"{self.dataset._input_text_affixes[0]}"
                f"{self.dataset._label_prefix}"
                f"{label}"
                f"{self.dataset._label_affix}"
            )
    
        # Add your query prefix (if any)
        prompt += self.dataset._query_prefix
    
        # Finally, append the query line without a label (just "sentiment: ")
        prompt += (
            f"{self.dataset._input_text_prefixes[0]}"
            f"{query_line[0]}"
            f"{self.dataset._input_text_affixes[0]}"
            f"{self.dataset._label_prefix}"
        )
    
        # For the trailing space after 'sentiment:' 
        prompt += " "
    
        return prompt

        
