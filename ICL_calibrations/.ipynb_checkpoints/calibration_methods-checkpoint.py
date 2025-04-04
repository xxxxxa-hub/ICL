from ICL_modules import s_random, functions

class calibration():
    def __init__(self) -> None:
        pass

    def train(self) -> None:
        pass

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        pass

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)

class domain_calibration(calibration):
    def __init__(self, label_space) -> None:
        self.label_space = label_space
        n_label = len(label_space)
        self.n_label = n_label
        self.calibrationA = [1e-5] * n_label

    def get_domain_sample(self, demonstration_set, sample_length):
        random = s_random.Random()
        while True:
            ret = []
            for i in range(len(demonstration_set[0][0])):
                output = []
                while len(output) < sample_length:
                    random_sample = demonstration_set[random.get_int_from_range(0, len(demonstration_set) - 1)][0][i]
                    random_sample = random_sample.split(' ')
                    random_index = random.get_int_from_range(0, len(random_sample) - 1)
                    output.append(random_sample[random_index])
                output = ' '.join(output)
                ret.append(output)
            yield ret

    def train(
        self,
        default_prompt_maker: callable, # input: demos_lines: <list[(list[str], str)]>, query_line: <list[str]> return: prompt, recommendation: prompt_writter.write_prompt_from_dataline
        feedforward: callable, # feedforward function, input: prompt: <str> return: label_space_prob
        demonstration_set = None,
        calibration_number = 20,
        sample_length = 64,
        k = 4,
        demonstration_set_index = None
    ) -> None:
        if demonstration_set_index is not None:
          demonstration_samples = demonstration_set_index
        else:
          random = s_random.Random()
          demonstration_samples = random.sample_index_set(calibration_number * k, len(demonstration_set), allow_repetition=True)
        gen = self.get_domain_sample(demonstration_set, sample_length)
        for i in range(calibration_number):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / calibration_number * 100),
                (i + 1),
                calibration_number
            ), ">>" * int((i + 1) / calibration_number * 32), end="")
            random_sentence = next(gen)

            prompt = default_prompt_maker([demonstration_set[demonstration_samples[j]] for j in range(k)], random_sentence)
            label_space_prob = feedforward(prompt = prompt, label_space = self.label_space)
            self.calibrationA = [self.calibrationA[j] + label_space_prob[j] for j in range(self.n_label)]
        self.calibrationA = [self.calibrationA[j] / calibration_number for j in range(self.n_label)]
        print("\nCalibration Training Finished.\n")

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return functions.softmax([label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label)])

class contextual_calibration(calibration):

    def __init__(self, label_space) -> None:
        self.label_space = label_space
        n_label = len(label_space)
        self.n_label = n_label
        self.calibrationA = [1e-5] * n_label

    def train(
        self,
        default_prompt_maker: callable, 
        feedforward: callable, 
        demonstration_set = None,
        k = 4,
        demonstration_set_index = None
    ) -> None:
        if demonstration_set_index is not None:
          demonstration_samples = demonstration_set_index
        else:
          my_random = stable_random.stable_random()
          demonstration_samples = my_random.sample_index_set(k, len(demonstration_set), allow_repetition=False)
        print(demonstration_samples)

        content_free = [[''],['NA'],['[MASK]']]
        for i, cf in enumerate(content_free):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / len(content_free) * 100),
                (i + 1),
                len(content_free)
            ), ">>" * int((i + 1) / len(content_free) * 32), end="")
            prompt = default_prompt_maker([demonstration_set[demonstration_samples[j]] for j in range(k)], cf)
            label_space_prob = feedforward(prompt = prompt, label_space = self.label_space)
            self.calibrationA = [self.calibrationA[j] + label_space_prob[j] for j in range(self.n_label)]
        self.calibrationA = [self.calibrationA[j] / len(content_free) for j in range(self.n_label)]
        print("\nCalibration Training Finished.\n")

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return functions.softmax([label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label)])

class lr_calib(calibration):

    def __init__(self, label_space, model, f_calibrate) -> None:
        self.label_space = label_space
        self.n_label = len(label_space)
        self.model = model
        self.failed = False
        self.f_calibrate = f_calibrate

    def permutate(self, elements, k):
        if k == 0:
            return [list(perm) for perm in itertools.permutations(elements, r=1)]
        else:
            extended_permutations = [
                list(base_perm + (extra_elem,))
                for base_perm in itertools.permutations(elements, r=k)
                for extra_elem in elements if extra_elem not in base_perm
            ]
            return extended_permutations

    def train(
        self,
        default_prompt_maker: callable,  
        feedforward: callable,             
        calibration_set=None,
        calibration_number=20,
        k=4,
        demonstration_set_index=None
    ):
        # Select indices for demonstration and query samples
        if demonstration_set_index is not None:
            demonstration_and_queue_samples = demonstration_set_index
        else:
            my_random = stable_random.stable_random()
            demonstration_and_queue_samples = my_random.sample_index_set(
                calibration_number, len(calibration_set), allow_repetition=False
            )
        print(demonstration_and_queue_samples)
        train_indexes = self.permutate(demonstration_and_queue_samples, k)

        probs = []
        labels = []
        demons = []
        #epsilon = 1e-9
        for i, ind in enumerate(train_indexes):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / len(train_indexes) * 100),
                (i + 1),
                calibration_number
            ), ">>" * int((i + 1) / len(train_indexes) * 32), end="")

            demonstration_samples = ind[:k]
            query_sample = ind[k]

            query = calibration_set[query_sample][0]
            label = calibration_set.get_label(query_sample)

            prompt = default_prompt_maker(
                demos_lines=[calibration_set[demonstration_samples[j]] for j in range(k)],
                query_line=query
            )
            label_space_probs = feedforward(prompt=prompt, label_space=self.label_space)
            probs.append(label_space_probs)
            labels.append(label)
            demons.append(demonstration_samples)

        # Construct training features:
        # For each probability vector pr = [P0, P1, ..., P_{n_label-1}],
        # compute features = [log(P1/P0), log(P2/P0), ..., log(P_{n_label-1}/P0)]
        X = np.array([
            [np.log((pr[i] ) / (pr[0] )) for i in range(1, self.n_label)]
            for pr in probs
        ])
        # Map labels to integer indices based on the provided label space
        y = np.array([self.label_space.index(l) for l in labels])

        # Create a DataFrame preserving the original structure
        df = pd.DataFrame({
            "label": y,
            "demons_index": demons,
            "features": [list(feat) for feat in X]
        })

        # Train the calibration model
        if self.f_calibrate:
            # Using a one-vs-all calibration scheme for multiclass with sigmoid calibration
            self.model = CalibratedClassifierCV(self.model, cv=3, method='sigmoid')
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        # Compute predictions and calibrated probabilities on the training set
        df["predicted"] = self.model.predict(X)
        calibrated_probs = self.model.predict_proba(X)
        for i in range(calibrated_probs.shape[1]):
            df[f"score_{i}"] = calibrated_probs[:, i]

        print("\nCalibration Training Finished.\n")
        return df

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        #epsilon = 1e-9
        # Compute (k-1)-dimensional feature vector using class 0 as the reference
        features = [np.log((label_space_prob[i] ) / (label_space_prob[0] ))
                    for i in range(1, self.n_label)]
        # Get calibrated probabilities for all classes
        calibrated_probs = self.model.predict_proba([features])[0]
        return list(calibrated_probs)

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)


def batch_calibration(
    label_space_probs: list[list[float]], 
    batch_size = 128, 
) -> list[list[float]]:
    ret = []
    step = len(label_space_probs) // batch_size
    for i in range(step):
        batch = label_space_probs[i * batch_size: (i + 1) * batch_size]
        mean_bias = [0] * len(batch[0])
        for j in range(batch_size):
            for k in range(len(batch[j])):
                mean_bias[k] += batch[j][k]
        mean_bias = [x / batch_size for x in mean_bias]
        for j in range(batch_size):
            ret.append(functions.softmax([batch[j][k] - mean_bias[k] for k in range(len(batch[j]))]))
    last_batch = label_space_probs[step * batch_size:]
    if len(last_batch) == 0:
        return ret
    mean_bias = [0] * len(last_batch[0])
    for j in range(len(last_batch)):
        for k in range(len(last_batch[j])):
            mean_bias[k] += last_batch[j][k]
    mean_bias = [x / len(last_batch) for x in mean_bias]
    for j in range(len(last_batch)):
        ret.append(functions.softmax([last_batch[j][k] - mean_bias[k] for k in range(len(last_batch[j]))]))
    return ret