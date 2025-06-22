from . import s_random
import pickle
import pkgutil
import copy
import random

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
        self._package_path = __package__[0:-5]

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
            with open("datasets/sst2.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["sentence"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._shuffle()
        
class agnews(datasets_loader):
    
    def __init__(self, from_cache = True):  
        super().__init__()

        self._input_text_prefixes = ["news: "]
        
        self._label_space = ['world', 'sports', 'business', 'sci/tech']
        self._ground_truth_label_space = ['world', 'sports', 'business', 'sci/tech']
        
        self._label_mapping = {0:0, 1:1, 2:2, 3:3} 
        self._label_prefix = "topic: "
        self.dataset_name = "AGNews" 
        
        self.input_element_numbers = 1
        
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "What is the topic of the news? ", "What is the news focused on? "],
            "input_text_prefixes": [["news: "], ["text: "], ["sentence: "]],
            "label_prefix": ["topic: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("glue", "sst2")['train']
            self._complie_dataset()
        else:
            with open("datasets/agnews.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)
            
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()
        
        
class rotten_tomatoes(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["review: "]
        self._label_space = ["negative", "positive"]
        self._label_prefix = "sentiment: "
        self._label_mapping = {0: 0, 1: 1}
        self.dataset_name = "rotten_tomatoes"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this review? ", "Please classify the sentiment of the following review. "],
            "input_text_prefixes": [["review: "], ["text: "], ["sentence: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("cornell-movie-review-data/rotten_tomatoes")['train']
            self._complie_dataset()
        else:
            with open("datasets/rotten_tomatoes.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()

class financial_phrasebank(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = ['negative', 'neutral', 'positive']
        self._label_prefix = "sentiment: "
        self._label_mapping = {0: 0, 1: 1, 2: 2}
        self.dataset_name = "financial_phrasebank"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "What is the attitude towards the financial news in this sentence? ",
                "What is the emotional response to the financial news in this sentence? "
            ],
            "input_text_prefixes": [["sentence: "], ["text: "], ["news: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")['train']
            self._complie_dataset()
        else:
            with open("datasets/financial_phrasebank.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["sentence"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()
class sst5(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        self._label_prefix = "sentiment: "
        self._label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.dataset_name = "SST5"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "How would you describe the overall feeling of the movie based on this sentence? ",
                "What mood does this sentence convey about the movie? "
            ],
            "input_text_prefixes": [["sentence: "], ["text: "], ["review: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("SetFit/sst5", "sentences_allagree")['train']
            self._complie_dataset()
        else:
            with open("datasets/sst5.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()
class trec(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["question: "]
        self._label_space = [
            'abbreviation',
            'entity',
            'description and abstract concept',
            'human being',
            'location',
            'numeric value'
        ]
        self._label_prefix = "target: "
        self._label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        self.dataset_name = "TREC"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "What is the topic of the question? ",
                "What is the primary focus of this question? "
            ],
            "input_text_prefixes": [["question: "], ["text: "], ["sentence: "]],
            "label_prefix": ["target: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("CogComp/trec")
            self._hgf_dataset = datasets.concatenate_datasets(
                [self._hgf_dataset['train'], self._hgf_dataset['test']]
            )
            self._complie_dataset()
        else:
            with open("datasets/trec.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["coarse_label"]))
        del self._hgf_dataset
        self._shuffle()


class subjective(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["review: "]
        self._label_space = ['objective', 'subjective']
        self._label_prefix = "subjectiveness: "
        self._label_mapping = {0: 0, 1: 1}
        self.dataset_name = "Subjective"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "Does this sentence reflect a personal opinion? ",
                "Is this sentence expressing a personal opinion or stating a fact? "
            ],
            "input_text_prefixes": [["review: "], ["text: "], ["sentence: "]],
            "label_prefix": ["subjectiveness: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("SetFit/subj")['train']
            self._complie_dataset()
        else:
            with open("datasets/subjective.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()


class tweet_eval_emotion(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["tweet: "]
        self._label_space = ['anger', 'joy', 'optimism', 'sadness']
        self._label_prefix = "emotion: "
        self._label_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        self.dataset_name = "tweet_eval_emotion"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "What feeling does this sentence convey? ",
                "What emotion does this sentence express? "
            ],
            "input_text_prefixes": [["tweet: "], ["text: "], ["sentence: "]],
            "label_prefix": ["emotion: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("tweet_eval", "emotion")
            self._hgf_dataset = datasets.concatenate_datasets([
                self._hgf_dataset['train'],
                self._hgf_dataset['validation'],
                self._hgf_dataset['test']
            ])
            self._complie_dataset()
        else:
            with open("datasets/tweet_eval_emotion.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()


class tweet_eval_hate(datasets_loader):
    def __init__(self, from_cache=True):
        super().__init__()

        self._input_text_prefixes = ["tweet: "]
        self._label_space = ['non-hate', 'hate']
        self._label_prefix = "hate speech: "
        self._label_mapping = {0: 0, 1: 1}
        self.dataset_name = "tweet_eval_hate"
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": [
                "",
                "Does this sentence contain hate speech? ",
                "Is this sentence an example of hate speech? "
            ],
            "input_text_prefixes": [["tweet: "], ["text: "], ["sentence: "]],
            "label_prefix": ["hate speech: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("tweet_eval", "hate")['train']
            self._complie_dataset()
        else:
            with open("datasets/tweet_eval_hate.dataset", "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._shuffle()


class hh_rlhf(datasets_loader):
    def __init__(self, from_cache=False, split="train"):
        super().__init__()

        self._input_text_prefixes = [""]
        self._input_text_affixes = [""] # End of the combined input string before label
        self._label_space = ["A", "B"]
        self._label_prefix = "" # Or "Preference: ", "Choose: "
        self._label_affix = "<|im_end|>\n"

        # Mapping for these two labels (0 for Option A, 1 for Option B)
        self._label_mapping = {0: 0, 1: 1}
        
        self.dataset_name = "hh-rlhf"
        self.input_element_numbers = 1 # Each example now has a single combined text input
        self.label_space_numbers = len(self._label_space) # Will be 2
        self._instruction = "<|im_start|>system\nYou are a helpful assistant that evaluates two responses to a user's query. Your task is to identify the preferred response by outputting only the corresponding letter, 'A' or 'B'.<|im_end|>\n"
        # The query will be embedded at the end of the combined input text,
        # so _query_prefix can remain empty or contain a final prompt like "Preferred Option: "
        self._query_prefix = "" 

        self.alternate_template = {
            "instruction": [
                "", # Empty instruction
                "Which of the two provided options in the following dialogue is better?",
                "Select the more helpful and harmless option for the given conversation."
            ],
            "input_text_prefixes": [
                ["Dialogue: "],
                ["Conversation: "],
                ["Text: "]
            ],
            "label_prefix": ["Preferred Option: ", "Choice: ", "Best Option: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("Anthropic/hh-rlhf")['train']
            self._complie_dataset(tokenizer)

            cache_file_name = f"datasets/hh_rlhf.dataset"
            pickle.dump(self.table, open(cache_file_name, "wb"))
        else:
            cache_file_name = f"datasets/hh_rlhf.dataset"
            with open(cache_file_name, "rb") as f:
                pickle_file = f.read()
            self.table = pickle.loads(pickle_file)


    def _complie_dataset(self, tokenizer):
        self.table = []
        breakpoint()
        for i in range(len(self._hgf_dataset)):
            original_chosen_text = self._hgf_dataset[i]["chosen"]
            original_rejected_text = self._hgf_dataset[i]["rejected"]
            
            formatted_chat_list, preferred_response = format_hh_row_as_chat_list(original_chosen_text, original_rejected_text)
            full_prompt_string = tokenizer.apply_chat_template(formatted_chat_list, tokenize=False, add_generation_prompt=True)
            
            system_end_pos = full_prompt_string.find("<|im_end|>\n")
            full_prompt_string_without_system = full_prompt_string[system_end_pos + len("<|im_end|>\n"):]
            
            self.table.append((
                [full_prompt_string_without_system], # Input is a list with one combined string
                preferred_response
            ))

        del self._hgf_dataset # Free up memory
        self._shuffle() # Shuffle the combined dataset

    def get_input_text(self, index: int) -> list[str]:
        # Returns a list containing one combined string (dialogue + options)
        return self.table[index][0]

    def get_label(self, index: int) -> str:
        # Returns either 'Option A' or 'Option B'
        return self.label_index_to_text(self.table[index][1])


def parse_hh_conversation(conversation_string):
    """
    Parses a full conversation string from the hh-rlhf dataset into a
    dialogue history and the final assistant response.
    (This function is unchanged from the previous example)
    """
    last_assistant_pos = conversation_string.rfind("\n\nAssistant:")
    if last_assistant_pos == -1:
        return [], ""

    context = conversation_string[:last_assistant_pos]
    final_response = conversation_string[last_assistant_pos + len("\n\nAssistant:"):].strip()
    
    dialogue_history = []
    turns = context.strip().split("\n\n")
    
    for turn in turns:
        if turn.startswith("Human:"):
            # The standard role for the user is 'user'
            role = "user"
            content = turn[len("Human:"):].strip()
            dialogue_history.append({"role": role, "content": content})
        elif turn.startswith("Assistant:"):
            # The standard role for the assistant is 'assistant'
            role = "assistant"
            content = turn[len("Assistant:"):].strip()
            dialogue_history.append({"role": role, "content": content})
            
    return dialogue_history, final_response


def format_hh_row_as_chat_list(chosen_str, rejected_str):
    """
    Takes a row from the hh-rlhf dataset and formats it into a list of
    dictionaries suitable for chat model fine-tuning.
    
    Randomly assigns chosen/rejected to A/B to prevent positional bias.
    """
    dialogue_history, chosen_response = parse_hh_conversation(chosen_str)
    _, rejected_response = parse_hh_conversation(rejected_str)
    
    # Randomize which response is A and which is B
    if random.choice([True, False]):
        option_a = chosen_response
        option_b = rejected_response
        preferred_response = 0
    else:
        option_a = rejected_response
        option_b = chosen_response
        preferred_response = 1

    # --- Assemble the final list of dictionaries ---
    
    messages = []
    # 2. Add the dialogue history (context)
    messages.extend(dialogue_history)
    
    # 3. Add the final user turn, which presents the choice
    final_user_content = (
        "The assistant provided two different responses to the last user query. Please identify which response is preferred.\n\n"
        f"Response A: {option_a}\n\n"
        f"Response B: {option_b}\n\n"
        "Preferred Response: "
    )
    messages.append({
        "role": "user",
        "content": final_user_content
    })
    
    return messages, preferred_response