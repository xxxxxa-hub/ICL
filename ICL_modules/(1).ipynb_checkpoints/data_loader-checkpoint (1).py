from . import s_random
import pickle
import copy

class datasets_loader():
    def __init__(self):
        self._hgf_dataset = None  # Huggingface Dataset Class. Will be overloaded by datasets.load_dataset.
        self._instruction = ""  # STRING. Instruction for the dataset in the begining of prompts. Can't be None.
        self._input_text_prefixes = ["Input: "] # LIST of STRING. Prefixes for the input text.
        self._input_text_affixes = [" "] # LIST of STRING. Affixes for the input text.
        self._label_prefix = "Label: " # STRING. Prefix for the label.
        self._label_affix = "\n" # STRING. Affix for the label.
        self._query_prefix = "" # STRING. Prefix for the query.
        self._label_space = [""] # LIST of STRING. Space for the label. Will be overloaded by the dataset.
        self._ground_truth_label_space = None # LIST of STRING. Ground truth label space. Will be overloaded by the dataset.
        self._reducted_label_space = None # LIST of STRING. Reducted label space. Will be overloaded by the dataset.
        self._label_mapping = {} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.table = None # LIST of (STRING, STRING). The table form of the dataset. Will be create by _transform_hgf_dataset_to_table.

    def _complie_dataset(self):
        # This function is used to transform the huggingface dataset to a table. And shuffle, cut the overlength data.
        # And also calculate the label_space_numbers and input_element_numbers.
        # Finally, delete the _hgf_dataset.
        pass

    def _shuffle(self):
        random = s_random.Random()
        index = random.sample_index_set(len(self), len(self))
        self.table = [self.table[i] for i in index]

    def __len__(self) -> int:
        # Should return the number of elements in the dataset.
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[list[str], str]:
        # Should return a (list of strings, string). 
        # list of string: The length is the number of input elements.
        # string: The label.
        return (self.get_input_text(index), self.get_label(index))
    def get_dataset(self):
        return self.table

    def get_input_text_prefixes(self):
        return self._input_text_prefixes
    
    def get_input_text_affixes(self):
        return self._input_text_affixes
    
    def get_label_prefix(self):
        return self._label_prefix
    
    def get_label_affix(self):
        return self._label_affix
    
    def get_instruction(self):
        return self._instruction
    
    def get_query_prefix(self):
        return self._query_prefix
    
    def get_label_space(self):
        return self._label_space
    
    def get_alternate_template(self):
        return self.alternate_template
        
    def get_input_text(self, index: int) -> list[str]:
        # Should return a list of strings. The length is the number of input elements.
        return self.table[index][0]

    def get_label(self, index: int) -> str:
        # Should return a string. Should call the _label_mapping.
        return self.label_index_to_text(self.table[index][1])
        

    def label_index_to_text(self, label_index: int) -> str:
        return copy.deepcopy(self._label_space[label_index])
    
    def find_index_from_label(self, label: str) -> int:
        # Should return the index of the label in the label space.
        return self._label_space.index(label)

    def split(self, split_indexes: list[list[int]]):
        ls = []
        for indexes in split_indexes:
            new_dataset = copy.deepcopy(self)
            new_dataset.table = [new_dataset.table[i] for i in indexes]
            ls.append(new_dataset)
        return ls

class glue_sst2(datasets_loader):

    def __init__(self,  from_cache = False):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = ["negative", "positive"] 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "sentiment: "
        #self._label_mapping = {0:0, 1:1}
        self.dataset_name = "GLUE-SST2" # STRING. Name of the dataset. Will be overloaded by the dataset.
        

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this sentence? ", "Please classify the sentiment of the following sentence. "],
            "input_text_prefixes": [["sentence: "], ["text: "], ["review: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("glue", "sst2")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/sst2.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/sst2.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["sentence"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._shuffle()