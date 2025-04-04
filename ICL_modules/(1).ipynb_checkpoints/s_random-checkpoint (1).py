import random

class Random:
    def __init__(self, seed=107):
        self._random = random.Random(seed)
    
    def get_float(self):
        """Returns a random float in the range [0.0, 1.0)."""
        return self._random.random()
    
    def get_int_from_range(self, start, end):
        """
        Returns a random integer N such that start <= N < end.
        """
        return self._random.randrange(start, end)
    
    def sample_one_element_from_list(self, lst):
        """
        Returns a single random element from the provided list.
        """
        return self._random.choice(lst)
    
    def sample_n_elements_from_list(self, lst, n, allow_repetition=False):
        """
        Returns a list of n elements sampled from lst.
        If allow_repetition is True, elements may be repeated; otherwise, they are unique.
        """
        if allow_repetition:
            return self._random.choices(lst, k=n)
        else:
            if n > len(lst):
                raise ValueError("n should be less than or equal to the length of the list when repetition is not allowed")
            return self._random.sample(lst, n)
    
    def sample_index_set(self, sample_number, max_index, allow_repetition=False):
        """
        Returns a list of sample_number indices chosen from the range [0, max_index).
        """
        indices = list(range(max_index))
        return self.sample_n_elements_from_list(indices, sample_number, allow_repetition)
    
    def shuffle_list(self, lst):
        """
        Returns a shuffled copy of the input list.
        """
        lst_copy = lst.copy()
        self._random.shuffle(lst_copy)
        return lst_copy
