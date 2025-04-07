from ICL_modules import s_random, functions
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import functools
import random
import copy
from scipy.optimize import minimize
from collections import defaultdict

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
                    if len(random_sample) > 1:
                        
                        random_index = random.get_int_from_range(0, len(random_sample) - 1)  # random.get_int_from_range(0,len(random_sample)-1)
                        output.append(random_sample[random_index]) # If random_sample has only one element or is empty, append the element if it exists or skip
                    elif len(random_sample) == 1:
                        output.append(random_sample[0])
                    
                    # random_index = random.get_int_from_range(0, len(random_sample) - 1)
                    # output.append(random_sample[random_index])
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
        
        demonstration_samples = demonstration_set_index
        
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
        total = sum(label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label))
        return [(label_space_prob[j] / self.calibrationA[j])/total for j in range(self.n_label)]  #functions.softmax([label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label)])

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

###########################################

class lr_calib_scipy_1d_cos(calibration):
    """
    This class implements an independent (univariate) calibration for each non-reference class.
    
    For a prediction with original probabilities [P(y=0|x), P(y=1|x), ..., P(y=n_label-1|x)],
    the features are computed as:
        x_c = log(P(y=c|x)/P(y=0|x)) for c = 1,...,n_label-1.
    
    The calibration equations are:
        log(P*(y=c|x)/P*(y=0|x)) = b_c + w_c * x_c, for c = 1,..., n_label-1,
    with the reference class fixed:
        logit_0 = 0.
    
    The calibrated probabilities are then given by:
        P*(y=0|x) = 1 / (1 + sum_{c=1}^{n_label-1} exp(b_c + w_c * x_c))
        P*(y=c|x) = exp(b_c + w_c * x_c) / (1 + sum_{j=1}^{n_label-1} exp(b_j + w_j * x_j)).
    
    A constraint is imposed on the calibration parameters: for each non-reference class c,
    we require the cosine similarity between [b_c, w_c] and [0, 1] (i.e. w_c/√(b_c²+w_c²)) to be high.
    Specifically, we require:
    
        (1/(n_label-1)) * sum_{c=1}^{n_label-1} (w_c / √(b_c²+w_c²)) >= cosine_threshold.
    """
    
    def __init__(
        self,
        label_space,
        use_invariance=True,
        lambda_invariance=1.0,
        invariance_loss_type='sym_ce',
        constraint=False,
        max_iter=100,
        verbose=False,
        k=None,
        dic=None,
        cosine_threshold=0.9  # minimum average cosine similarity required
    ):
        super().__init__()
        self.label_space = label_space
        self.n_label = len(label_space)
        self.use_invariance = use_invariance
        self.lambda_invariance = lambda_invariance
        self.invariance_loss_type = invariance_loss_type
        self.constraint = constraint
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Additional parameters
        self.k = k
        self.dic = dic
        self.cosine_threshold = cosine_threshold
        
        # Final learned parameters.
        # We only optimize for non-reference classes (classes 1,..., n_label-1).
        # For each such class we have 2 parameters: [b_c, w_c].
        # Total parameters = (n_label - 1) * 2.
        self.params_ = None
        self.fitted_ = False

    # -------------------------------------------------------------------------
    # Feature building:
    # For a given probability vector, compute x = [log(P(y=1)/P(y=0)), ..., log(P(y=n_label-1)/P(y=0))]
    # -------------------------------------------------------------------------
    def _make_features(self, prob_vector):
        eps = 1e-9
        base = prob_vector[0] + eps
        feats = []
        for c in range(1, self.n_label):
            feats.append(np.log(prob_vector[c] / base))
        return np.array(feats, dtype=float)  # shape: (n_label-1,)

    # -------------------------------------------------------------------------
    # Unpack parameters:
    # The optimized parameters (a 1D array of length (n_label-1)*2) are reshaped into a matrix.
    # Row c (for c=1,..., n_label-1) contains [b_c, w_c].
    # For the reference class (class 0) we set parameters to zeros.
    # -------------------------------------------------------------------------
    def _unpack_params(self, params):
        non_ref = params.reshape(self.n_label - 1, 2)  # shape: (n_label-1, 2)
        ref = np.zeros((1, 2))
        param_matrix = np.vstack([ref, non_ref])
        return param_matrix  # shape: (n_label, 2)

    # -------------------------------------------------------------------------
    # Compute logits:
    # For each sample with feature vector x (length n_label-1),
    # the logit for class 0 is 0, and for class c (c>=1):
    # logit_c = b_c + w_c * x[c-1]
    # -------------------------------------------------------------------------
    def _compute_logits(self, param_matrix, x):
        logits = [0.0]  # class 0
        for c in range(1, self.n_label):
            b_c = param_matrix[c, 0]
            w_c = param_matrix[c, 1]
            logits.append(b_c + w_c * x[c - 1])
        return np.array(logits)

    # -------------------------------------------------------------------------
    # Negative Log-Likelihood (using softmax)
    # -------------------------------------------------------------------------
    def _negative_log_likelihood(self, params, X, Y):
        param_matrix = self._unpack_params(params)
        eps = 1e-9
        N = len(X)
        ll = 0.0
        for i in range(N):
            logits_i = self._compute_logits(param_matrix, X[i])
            shift = logits_i - np.max(logits_i)
            exps = np.exp(shift)
            sumExps = np.sum(exps)
            prob = exps / (sumExps + eps)
            y_i = int(Y[i])
            ll -= np.log(prob[y_i] + eps)
        return ll

    # -------------------------------------------------------------------------
    # Invariance penalty (as in previous models)
    # -------------------------------------------------------------------------
    def _invariance_penalty(self, params, X, pairs):
        param_matrix = self._unpack_params(params)
        eps = 1e-9
        N = len(X)
        all_probs = []
        for i in range(N):
            logits_i = self._compute_logits(param_matrix, X[i])
            shift = logits_i - np.max(logits_i)
            exps = np.exp(shift)
            sumExps = np.sum(exps)
            p_i = exps / (sumExps + eps)
            all_probs.append(p_i)
        total_pen = 0.0
        for (i, j) in pairs:
            p_i = all_probs[i]
            p_j = all_probs[j]
            if self.invariance_loss_type == 'mse':
                total_pen += np.sum((p_i - p_j) ** 2)
            elif self.invariance_loss_type == 'l1':
                total_pen += np.sum(np.abs(p_i - p_j))
            elif self.invariance_loss_type == 'sym_ce':
                ce_ij = -np.sum(p_j * np.log(p_i + eps))
                ce_ji = -np.sum(p_i * np.log(p_j + eps))
                total_pen += (ce_ij + ce_ji)
            else:
                total_pen += np.sum((p_i - p_j) ** 2)
        return total_pen

    # -------------------------------------------------------------------------
    # Full objective: NLL + lambda_invariance * invariance penalty
    # -------------------------------------------------------------------------
    def _objective(self, params, X, Y, pairs):
        nll = self._negative_log_likelihood(params, X, Y)
        if self.use_invariance and len(pairs) > 0:
            pen = self._invariance_penalty(params, X, pairs)
        else:
            pen = 0.0
        return nll + self.lambda_invariance * pen

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        default_prompt_maker: callable,
        feedforward: callable,
        demonstration_set=None,
        k=4,
        demonstration_set_index=None
    ):
        print(demonstration_set_index)
        train_indexes = self._permutate(demonstration_set_index, k)

        probs_list = []
        labels_list = []
        queries_list = []

        total = len(train_indexes)
        for i, ind in enumerate(train_indexes):
            print(
                f"\rProcess: {int((i + 1) / total * 100)}% " +
                f"[{'>>' * int((i + 1) / total * 32)}" +
                f"{'.' * (32 - int((i + 1) / total * 32))}] " +
                f"{i+1}/{total}",
                end="", flush=True
            )
            demonstration_samples = ind[:k]
            query_sample = ind[k]
            query = demonstration_set[query_sample][0]
            label = demonstration_set.get_label(query_sample)
            prompt = default_prompt_maker(
                [demonstration_set[demonstration_samples[j]] for j in range(k)],
                query
            )
            label_space_probs = feedforward(prompt=prompt, label_space=self.label_space)
            probs_list.append(label_space_probs)
            labels_list.append(label)
            queries_list.append(query_sample)
        print()

        # Build features and labels.
        X_list = []
        y_list = []
        for pr, lab in zip(probs_list, labels_list):
            x_vec = self._make_features(pr)  # shape: (n_label-1,)
            X_list.append(x_vec)
            y_list.append(self.label_space.index(lab))
        X = np.array(X_list, dtype=float)
        Y = np.array(y_list, dtype=float)
        N = len(X)

        df = pd.DataFrame({
            "label": Y,
            "query_index": queries_list,
            "features": list(X_list)
        })

        # Build pairs for invariance penalty.
        query_map = defaultdict(list)
        for i, qid in enumerate(df["query_index"]):
            query_map[qid].append(i)
        pairs = []
        for qid, idxs in query_map.items():
            if len(idxs) < 2:
                continue
            for i1 in range(len(idxs)):
                for i2 in range(i1 + 1, len(idxs)):
                    pairs.append((idxs[i1], idxs[i2]))

        # Prepare initial parameters: shape = ((n_label-1)*2,)
        n_dim = 1  # each non-reference calibrator is univariate.
        init_params = np.zeros((self.n_label - 1) * 2, dtype=float)

        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        # -------------------------------
        # BUILD CONSTRAINTS (cosine similarity constraint)
        # -------------------------------
        constraints_list = []
        if self.constraint:
            def cosine_constraint(params):
                # Unpack: each non-reference class c has parameters [b_c, w_c]
                non_ref = params.reshape(self.n_label - 1, 2)
                cosine_sum = 0.0
                for c in range(self.n_label - 1):
                    b_c, w_c = non_ref[c]
                    norm = np.sqrt(b_c**2 + w_c**2)
                    if norm < 1e-9:
                        cosine = 0.0
                    else:
                        cosine = w_c / norm  # cosine similarity with [0,1]
                    cosine_sum += cosine
                avg_cosine = cosine_sum / (self.n_label - 1)
                return avg_cosine - self.cosine_threshold
            constraints_list.append({'type': 'ineq', 'fun': cosine_constraint})
            solver_method = 'trust-constr'
        else:
            solver_method = 'BFGS'

        # -------------------------------
        # Run optimization.
        # -------------------------------
        res = minimize(
            func_to_minimize,
            init_params,
            method=solver_method,
            constraints=constraints_list,
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )

        self.params_ = res.x
        self.fitted_ = True

        if self.verbose:
            print("\n[SCIPY] Optimization success:", res.success)
            print("[SCIPY] Message:", res.message)

        # Compute predictions on the training set.
        param_matrix = self._unpack_params(self.params_)
        all_probs = []
        for i in range(N):
            logits_i = self._compute_logits(param_matrix, X[i])
            shift = logits_i - np.max(logits_i)
            exps = np.exp(shift)
            sumExps = np.sum(exps)
            p_i = exps / (sumExps + 1e-9)
            all_probs.append(p_i)
        for c in range(self.n_label):
            df[f"score_{c}"] = [p[c] for p in all_probs]
        df["predicted"] = [np.argmax(p) for p in all_probs]

        print("\n[SCIPY] Calibration Training Finished.\n")
        return df

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def inference(self, label_space_prob, full_vocab_prob, hidden_state):
        if not self.fitted_:
            raise RuntimeError("lr_calib_scipy model not fitted yet.")
        x = self._make_features(label_space_prob)
        param_matrix = self._unpack_params(self.params_)
        logits = self._compute_logits(param_matrix, x)
        shift = logits - np.max(logits)
        exps = np.exp(shift)
        sumExps = np.sum(exps)
        final_probs = exps / (sumExps + 1e-9)
        return final_probs.tolist()

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state):
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)

    # -------------------------------------------------------------------------
    # Permutation helper
    # -------------------------------------------------------------------------
    def _permutate(self, elements, k):
        if k == 0:
            return [list(perm) for perm in itertools.permutations(elements, r=1)]
        else:
            extended_permutations = [
                list(base_perm + (extra_elem,))
                for base_perm in itertools.permutations(elements, r=k)
                for extra_elem in elements if extra_elem not in base_perm
            ]
            return extended_permutations    
        



