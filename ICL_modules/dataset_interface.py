from . import s_random
import random
from . import data_loader
import copy
from typing import List


class DatasetSplitter:
    def __init__(self, dataset: data_loader.datasets_loader, demonstration_number, test_number, random_seed, permute_demonstrations=True):
        self.dataset = dataset
        self.random_seed = random_seed
        self.demonstration_number = demonstration_number
        self.test_number = test_number
        self.dataset_name = dataset.dataset_name
        self.permute_demonstrations = permute_demonstrations

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
            

    def prompt_writter(self, demonstrations: List[tuple] , query_line):
        """
        Builds a prompt by iterating over demonstrations and then appending
        the query line. Incorporates all dataset-specific prefixes/affixes.
        """
        # Special handling for hh_rlhf dataset
        if self.dataset.get_conversational():
            if self.dataset.dataset_name == "hh_rlhf":
                return self._build_conversational_messages(demonstrations, query_line, True)
            elif self.dataset.dataset_name == "ultrafeedback":
                return self._build_conversational_messages(demonstrations, query_line, False)
        
        # Original code for other datasets
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
        # Do not add space for hh-rlhf
        # prompt += " "
    
        return prompt

    def _build_conversational_messages(self, demonstrations: List[tuple], query_dict: dict, multi_round: bool = True) -> List[dict]:
        """
        Build messages list for hh_rlhf dataset.
        demonstrations: List of (prompt_dict, label) pairs
        query_line: A prompt_dict
        multi_round: If True, use multi-round conversation format. If False, use single-round format from SimPO.
        """
        messages = []

        if multi_round:
            # Predict A/B
            system_prompt = (
                "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
                "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
                "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were "
                "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
                "of the assistants. Be as objective as possible. After careful consideration, output your final verdict by strictly following this format: "
                '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better. '
                'Output **only** "[[A]]" or "[[B]]" and nothing else.'
            )

            messages.append({"role": "system", "content": system_prompt})

            # Collect all demonstration blocks
            demo_blocks = []
            for prompt_dict, label in demonstrations:
                # First demonstration: option_a as Assistant A, option_b as Assistant B
                demo_messages_1 = prompt_dict["dialogue_history"].copy()

                if demo_messages_1:
                    last_message_1 = demo_messages_1[-1].copy()
                    last_message_1["content"] += f"\n\n[The Start of Assistant A's Answer]\n{prompt_dict['option_a']}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{prompt_dict['option_b']}\n[The End of Assistant B's Answer]"
                    demo_messages_1[-1] = last_message_1

                    # Create first demo block
                    block_1 = demo_messages_1 + [{"role": "assistant", "content": f"[[{label}]]"}]
                    demo_blocks.append(block_1)

                # Only create permuted version if permute_demonstrations is True
                if self.permute_demonstrations:
                    # Second demonstration: option_b as Assistant A, option_a as Assistant B (swapped)
                    demo_messages_2 = prompt_dict["dialogue_history"].copy()

                    if demo_messages_2:
                        last_message_2 = demo_messages_2[-1].copy()
                        last_message_2["content"] += f"\n\n[The Start of Assistant A's Answer]\n{prompt_dict['option_b']}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{prompt_dict['option_a']}\n[The End of Assistant B's Answer]"
                        demo_messages_2[-1] = last_message_2

                        # Create second demo block with flipped label
                        # flipped_label = "No" if label == "Yes" else "Yes"
                        flipped_label = "B" if label == "A" else "A"
                        block_2 = demo_messages_2 + [{"role": "assistant", "content": f"[[{flipped_label}]]"}]
                        demo_blocks.append(block_2)

            # Shuffle the demonstration blocks
            random.shuffle(demo_blocks)

            # Add shuffled demonstration blocks to messages with example indicators
            for i, block in enumerate(demo_blocks, 1):
                # Add "Example {i}\n" to the first user message content in each block
                block_with_example = block.copy()
                for j, msg in enumerate(block_with_example):
                    if msg["role"] == "user":
                        msg_copy = msg.copy()
                        msg_copy["content"] = f"Example {i}\n{msg['content']}"
                        block_with_example[j] = msg_copy
                        break
                messages.extend(block_with_example)

            # Add query
            query_messages = query_dict["dialogue_history"].copy()

            # Append options to the last query message content
            if query_messages:
                last_query_message = query_messages[-1].copy()
                last_query_message["content"] += f"\n\n[The Start of Assistant A's Answer]\n{query_dict['option_a']}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{query_dict['option_b']}\n[The End of Assistant B's Answer]"
                query_messages[-1] = last_query_message

                # Add "Example {i}\n" to the first user message in the query
                query_example_num = len(demo_blocks) + 1
                for j, msg in enumerate(query_messages):
                    if msg["role"] == "user":
                        msg_copy = msg.copy()
                        msg_copy["content"] = f"Example {query_example_num}\n{msg['content']}"
                        query_messages[j] = msg_copy
                        break

                # Add query messages
                messages.extend(query_messages)

        else:
            # Single-round format from SimPO
            system_prompt = (
                "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
                "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
                "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were "
                "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
                "of the assistants. Be as objective as possible. After careful consideration, output your final verdict by strictly following this format: "
                '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better. '
                'Output **only** "[[A]]" or "[[B]]" and nothing else.'
            )

            messages.append({"role": "system", "content": system_prompt})

            # Collect demonstration pairs for single-round format
            demo_pairs = []
            for prompt_dict, label in demonstrations:
                # Extract question from dialogue_history (only one dict when multi_round=False)
                question = prompt_dict["dialogue_history"][0]["content"]

                # First demonstration: option_a as Assistant A, option_b as Assistant B
                user_content_1 = f"""Question: {question}

[The Start of Assistant A's Answer]
{prompt_dict['option_a']}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{prompt_dict['option_b']}
[The End of Assistant B's Answer]"""

                demo_pairs.append(({"role": "user", "content": user_content_1}, {"role": "assistant", "content": f"[[{label}]]"}))

                # Only create permuted version if permute_demonstrations is True
                if self.permute_demonstrations:
                    # Second demonstration: option_b as Assistant A, option_a as Assistant B (swapped)
                    user_content_2 = f"""Question: {question}

[The Start of Assistant A's Answer]
{prompt_dict['option_b']}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{prompt_dict['option_a']}
[The End of Assistant B's Answer]"""

                    flipped_label = "B" if label == "A" else "A"
                    demo_pairs.append(({"role": "user", "content": user_content_2}, {"role": "assistant", "content": f"[[{flipped_label}]]"}))

            # Shuffle the demonstration pairs
            random.shuffle(demo_pairs)

            # Add shuffled demonstrations to messages
            for user_msg, assistant_msg in demo_pairs:
                messages.append(user_msg)
                messages.append(assistant_msg)

            # Add the current query
            query_question = query_dict["dialogue_history"][0]["content"]
            query_content = f"""Question: {query_question}

[The Start of Assistant A's Answer]
{query_dict['option_a']}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{query_dict['option_b']}
[The End of Assistant B's Answer]"""

            messages.append({"role": "user", "content": query_content})

        return messages

    def _format_hh_conversation(self, conv_data: dict) -> str:
        """
        Format hh_rlhf conversation data into a string for prompt building.
        """
        formatted_parts = []
        
        # Add dialogue history
        for turn in conv_data["dialogue_history"]:
            role = turn["role"].capitalize()
            content = turn["content"]
            formatted_parts.append(f"{role}: {content}")
        
        # Add the two response options
        formatted_parts.append(f"\nThe assistant provided two different responses to the last user query. Please identify which response is preferred.")
        formatted_parts.append(f"\nResponse A: {conv_data['option_a']}")
        formatted_parts.append(f"\nResponse B: {conv_data['option_b']}")
        formatted_parts.append(f"\nPreferred Response: ")
        
        return "\n\n".join(formatted_parts)

        
