import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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




class lr_calib_scipy(calibration):
    """
    This class uses scipy.optimize.minimize to fit a logistic regression model
    with an invariance penalty across different contexts for the same query.

    Supported penalty types: 'mse', 'l1', 'sym_ce'.
    """

    def __init__(
        self,
        label_space,
        use_invariance=True,
        lambda_invariance=1.0,
        invariance_loss_type='mse',
        max_iter=100,
        verbose=False
    ):
        super().__init__()
        self.label_space = label_space
        self.n_label = len(label_space)
        self.use_invariance = use_invariance
        self.lambda_invariance = lambda_invariance
        self.invariance_loss_type = invariance_loss_type
        self.max_iter = max_iter
        self.verbose = verbose

        # We'll store the final b0, b1 after fitting
        self.b0_ = 0.0
        self.b1_ = 0.0
        self.fitted_ = False

    def _logistic(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _negative_log_likelihood(self, params, X, Y):
        """
        Standard binary logistic regression negative log-likelihood:
          - sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]
        """
        b0, b1 = params
        logits = b0 + b1 * X
        probs = self._logistic(logits)
        eps = 1e-9
        ll = -np.sum(Y * np.log(probs + eps) + (1 - Y) * np.log(1 - probs + eps))
        return ll

    def _invariance_penalty(self, params, X, pairs):
        """
        Invariance penalty across pairs (i, j) that share the same query.
        pairs: list of (i, j) with i < j
        """
        b0, b1 = params
        logits = b0 + b1 * X
        probs = self._logistic(logits)
        eps = 1e-9
        total_pen = 0.0

        for i, j in pairs:
            p_i = probs[i]
            p_j = probs[j]
            if self.invariance_loss_type == 'mse':
                total_pen += (p_i - p_j) ** 2
            elif self.invariance_loss_type == 'l1':
                total_pen += abs(p_i - p_j)
            elif self.invariance_loss_type == 'sym_ce':
                ce_ij = -(p_j * np.log(p_i + eps) + (1 - p_j) * np.log(1 - p_i + eps))
                ce_ji = -(p_i * np.log(p_j + eps) + (1 - p_i) * np.log(1 - p_j + eps))
                total_pen += (ce_ij + ce_ji)
            else:
                # default to MSE
                total_pen += (p_i - p_j) ** 2

        return total_pen

    def _objective(self, params, X, Y, pairs):
        """
        Full objective = NLL + alpha * InvariancePenalty
        """
        nll = self._negative_log_likelihood(params, X, Y)
        if self.use_invariance and len(pairs) > 0:
            pen = self._invariance_penalty(params, X, pairs)
        else:
            pen = 0.0
        return nll + self.lambda_invariance * pen

    def train(
        self,
        default_prompt_maker: callable,
        feedforward: callable,
        demonstration_set=None,
        k=4,
        demonstration_set_index=None
    ):
        """
        Similar to your original 'train' method, but we'll do SciPy-based optimization.
        1) Build the DataFrame 'df'
        2) Construct pairs
        3) Optimize logistic regression
        4) Return the DataFrame with predictions
        """
        print(demonstration_set_index)
        # 1. Build the training set
        train_indexes = self._permutate(demonstration_set_index, k)

        probs = []
        labels = []
        queries = []

        total = len(train_indexes)
        for i, ind in enumerate(train_indexes):
            print(
                f"\rProcess: {int((i + 1) / total * 100)}% "
                f"[{'>>' * int((i + 1) / total * 32)}"
                f"{'.' * (32 - int((i + 1) / total * 32))}] "
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
            probs.append(label_space_probs)
            labels.append(label)
            queries.append(query_sample)

        # 2. For binary classification, we have [P0, P1]
        #    feature = log(P1 / P0)
        X_list = []
        y_list = []
        for pr, lab in zip(probs, labels):
            feature_val = np.log(pr[1] / pr[0])
            X_list.append(feature_val)
            y_list.append(self.label_space.index(lab))

        # Build a DataFrame
        df = pd.DataFrame({
            "label": y_list,
            "query_index": queries,
            "features": X_list
        })
        print()

        # 3. Build pairs for invariance
        query_map = defaultdict(list)
        for i, qid in enumerate(df["query_index"]):
            query_map[qid].append(i)
        pairs = []
        for qid, idxs in query_map.items():
            if len(idxs) < 2:
                continue
            for i1 in range(len(idxs)):
                for i2 in range(i1+1, len(idxs)):
                    pairs.append((idxs[i1], idxs[i2]))

        X = np.array(df["features"].values, dtype=float)
        Y = np.array(df["label"].values, dtype=float)

        # 4. SciPy minimization
        init_params = np.array([0.0, 0.0])  # b0, b1
        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        res = minimize(
            func_to_minimize,
            init_params,
            method='BFGS',
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )
        self.b0_, self.b1_ = res.x
        self.fitted_ = True
        if self.verbose:
            print("\n[SCIPY] Optimization success:", res.success)
            print("[SCIPY] Message:", res.message)
            print(f"[SCIPY] b0={self.b0_:.4f}, b1={self.b1_:.4f}")

        # 5. Compute predictions on the training set
        logits = self.b0_ + self.b1_ * X
        pred_probs = 1.0 / (1.0 + np.exp(-logits))
        df["score_1"] = pred_probs
        df["score_0"] = 1.0 - pred_probs
        df["predicted"] = (pred_probs > 0.5).astype(int)

        print("\n[SCIPY] Calibration Training Finished.\n")
        return df

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        """
        Inference using the learned b0, b1 from SciPy optimization.
        """
        if not self.fitted_:
            raise RuntimeError("lr_calib_scipy model not fitted yet.")
        # For binary classification:
        feature_val = np.log(label_space_prob[1] / label_space_prob[0])
        logit = self.b0_ + self.b1_ * feature_val
        prob1 = 1.0 / (1.0 + np.exp(-logit))
        prob0 = 1.0 - prob1
        return [prob0, prob1]

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)

    # -------------
    # HELPER FUNCS
    # -------------
    def _permutate(self, elements, k):
        """
        Duplicate of your permutate logic from lr_calib.
        """
        if k == 0:
            return [list(perm) for perm in itertools.permutations(elements, r=1)]
        else:
            extended_permutations = [
                list(base_perm + (extra_elem,))
                for base_perm in itertools.permutations(elements, r=k)
                for extra_elem in elements if extra_elem not in base_perm
            ]
            return extended_permutations
            
########################################################
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from scipy.optimize import minimize

# class calibration: ... # base class if needed

class lr_calib_scipy_2(calibration):
    """
    Merged class combining:
      - Multi-class logistic regression with invariance penalty
      - Optional constraint for both binary and multi-class
        * Binary (n_label=2): scale_factor * |(w1 - w0)| - |(b1 - b0)| > 0
          (which matches the original idea that (b0, b1) were differences)
        * Multi-class (n_label>2): For each class c, scale_factor * ||w_c||_2 - |b_c| > 0
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
        dic=None
    ):
        super().__init__()
        # Fields from your previous code
        self.label_space = label_space
        self.n_label = len(label_space)
        self.use_invariance = use_invariance
        self.lambda_invariance = lambda_invariance
        self.invariance_loss_type = invariance_loss_type
        self.constraint = constraint
        self.max_iter = max_iter
        self.verbose = verbose

        # Additional from "Class #1"
        self.k = k
        self.dic = dic

        # Final learned parameters
        # For n_label classes, param_matrix has shape (n_label, n_dim+1),
        # which we flatten into a 1D array. We'll store it in self.params_.
        self.params_ = None
        self.fitted_ = False

    # -------------------------------------------------------------------------
    #  Feature building and multi-class logistic
    # -------------------------------------------------------------------------
    def _make_features(self, prob_vector):
        """
        Convert raw probability vector (size n_label) into a feature vector.
        By default, we do log-ratios wrt prob[0]:
           X[i] = log(prob[i+1] / prob[0]) for i=0..(n_label-2).
        If n_label=2, that yields a single feature [log(prob[1]/prob[0])].
        """
        eps = 1e-9
        base = prob_vector[0] + eps
        feats = []
        for i in range(1, self.n_label):
            feats.append(np.log(prob_vector[i] / base))
        return np.array(feats, dtype=float)

    def _unpack_params(self, params):
        """
        Reshape the 1D 'params' array into (n_label, n_dim+1).
         - param_matrix[c, 0] is bias b_c
         - param_matrix[c, 1:] is weight vector w_c
        """
        n_dim = self.n_label - 1
        param_matrix = params.reshape(self.n_label, n_dim + 1)
        return param_matrix

    def _compute_logits(self, param_matrix, x):
        """
        Given a feature vector x (shape [n_dim]) and param_matrix
        (shape [n_label, n_dim+1]), compute logits for each class c.
        """
        logits = []
        for c in range(self.n_label):
            b_c = param_matrix[c, 0]
            w_c = param_matrix[c, 1:]
            logit_c = b_c + np.dot(w_c, x)
            logits.append(logit_c)
        return np.array(logits)

    def _negative_log_likelihood(self, params, X, Y):
        """
        Multi-class negative log-likelihood (cross-entropy).
         - X is shape [N, n_dim]
         - Y is shape [N], integer labels in [0..n_label-1]
        """
        param_matrix = self._unpack_params(params)
        eps = 1e-9
        N = len(X)
        ll = 0.0

        for i in range(N):
            logits_i = self._compute_logits(param_matrix, X[i])
            # softmax
            shift = logits_i - np.max(logits_i)
            exps = np.exp(shift)
            sumExps = np.sum(exps)
            prob = exps / (sumExps + eps)

            y_i = int(Y[i])
            ll -= np.log(prob[y_i] + eps)
        return ll

    def _invariance_penalty(self, params, X, pairs):
        """
        Invariance penalty across pairs (i, j).
        We'll compute the predicted distribution for each sample,
        then measure differences (MSE, L1, or symmetrical CE).
        """
        param_matrix = self._unpack_params(params)
        eps = 1e-9
        N = len(X)

        # Precompute probabilities for each sample
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
                total_pen += np.sum((p_i - p_j)**2)
            elif self.invariance_loss_type == 'l1':
                total_pen += np.sum(np.abs(p_i - p_j))
            elif self.invariance_loss_type == 'sym_ce':
                ce_ij = -np.sum(p_j * np.log(p_i + eps))
                ce_ji = -np.sum(p_i * np.log(p_j + eps))
                total_pen += (ce_ij + ce_ji)
            else:
                # default to MSE
                total_pen += np.sum((p_i - p_j)**2)

        return total_pen

    def _objective(self, params, X, Y, pairs):
        """
        Full objective = NLL + lambda_invariance * InvariancePenalty
        """
        nll = self._negative_log_likelihood(params, X, Y)
        if self.use_invariance and len(pairs) > 0:
            pen = self._invariance_penalty(params, X, pairs)
        else:
            pen = 0.0
        return nll + self.lambda_invariance * pen

    # -------------------------------------------------------------------------
    #  TRAIN
    # -------------------------------------------------------------------------
    def train(
        self,
        default_prompt_maker: callable,
        feedforward: callable,
        demonstration_set=None,
        k=4,
        demonstration_set_index=None
    ):
        """
        1) Build the training set (like in your old code):
           - Permute demonstration_set_index with _permutate
           - For each sample, call feedforward(...) to get label_space_probs
           - Convert that to features (log-ratios)
        2) Build pairs for invariance
        3) (Optional) apply your domain-specific "adaptive" lambda_invariance if binary
        4) Setup constraints (binary vs. multi-class)
        5) Minimize with SciPy
        6) Return df with predicted probabilities
        """
        print(demonstration_set_index)
        train_indexes = self._permutate(demonstration_set_index, k)

        probs_list = []
        labels_list = []
        queries_list = []

        total = len(train_indexes)
        for i, ind in enumerate(train_indexes):
            print(
                f"\rProcess: {int((i + 1) / total * 100)}% "
                f"[{'>>' * int((i + 1) / total * 32)}"
                f"{'.' * (32 - int((i + 1) / total * 32))}] "
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

        # Build the feature matrix X and label vector Y
        X_list = []
        y_list = []
        for pr, lab in zip(probs_list, labels_list):
            x_vec = self._make_features(pr)  # shape [n_dim = n_label-1]
            X_list.append(x_vec)
            # Convert label to integer index
            y_list.append(self.label_space.index(lab))

        X = np.array(X_list, dtype=float)   # shape [N, n_dim]
        Y = np.array(y_list, dtype=float)   # shape [N]
        N = len(X)

        # Build a DataFrame for debugging / final output
        df = pd.DataFrame({
            "label": Y,
            "query_index": queries_list,
            "features": list(X_list)
        })

        # Build pairs for invariance
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

        # (Optional) "adaptive" lambda_invariance if n_label=2
        if self.n_label == 2:
            zero_idx = (df["label"] == 0)
            one_idx  = (df["label"] == 1)
            if zero_idx.any() and one_idx.any():
                x_zero = [x[0] for x, lbl in zip(X, Y) if lbl == 0]
                x_one  = [x[0] for x, lbl in zip(X, Y) if lbl == 1]
                if len(x_zero) > 0 and len(x_one) > 0:
                    max_0 = max(x_zero)
                    min_1 = min(x_one)
                    if (min_1 - max_0) < 0:
                        self.lambda_invariance = 0.4

        # Prepare initial parameters
        # param_matrix shape = (n_label, n_dim+1)
        n_dim = self.n_label - 1
        init_params = np.zeros(self.n_label * (n_dim + 1), dtype=float)

        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        # -------------------------------
        #  BUILD CONSTRAINTS (binary vs. multi-class)
        # -------------------------------
        constraints_list = []

        # We'll define a scale factor from self.dic[self.k] if it exists
        scale_factor = 1.0
        if self.dic is not None and (self.k in self.dic):
            scale_factor = self.dic[self.k]

        if self.constraint:
            if self.n_label == 2:
                # ---------------------------------------------------------
                # Corrected Binary Constraint:
                # We interpret param_matrix => shape(2,2) => flatten => [b0, w0, b1, w1]
                # But the "difference" is: b_diff = b1 - b0, w_diff = w1 - w0
                # so constraint => scale_factor * |w_diff| - |b_diff| > 0
                # This matches your original logic where "b0, b1" were differences.
                # ---------------------------------------------------------
                def constraint_fun_binary(p):
                    # p = [b0, w0, b1, w1]
                    b_diff = p[2] - p[0]
                    w_diff = p[3] - p[1]
                    return scale_factor * abs(w_diff) - abs(b_diff)

                constraints_list = [{'type': 'ineq', 'fun': constraint_fun_binary}]

                solver_method = 'trust-constr'
            else:
                # Multi-class approach: For each class c, enforce
                # scale_factor * ||w_c||_2 - |b_c| > 0
                def constraint_fun_c(params, c):
                    param_matrix = self._unpack_params(params)
                    b_c = param_matrix[c, 0]
                    w_c = param_matrix[c, 1:]
                    return scale_factor * np.linalg.norm(w_c, 2) - abs(b_c)

                for c in range(self.n_label):
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': (lambda p, c=c: constraint_fun_c(p, c))
                    })
                solver_method = 'trust-constr'
        else:
            solver_method = 'BFGS'

        # -------------------------------
        #  RUN SCIPY MINIMIZATION
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

        # -------------------------------
        #  Compute predictions on the training set
        # -------------------------------
        param_matrix = self._unpack_params(self.params_)
        all_probs = []
        for i in range(N):
            logits_i = self._compute_logits(param_matrix, X[i])
            shift = logits_i - np.max(logits_i)
            exps = np.exp(shift)
            sumExps = np.sum(exps)
            p_i = exps / (sumExps + 1e-9)
            all_probs.append(p_i)

        # Store per-class scores in df
        for c in range(self.n_label):
            df[f"score_{c}"] = [p[c] for p in all_probs]
        df["predicted"] = [np.argmax(p) for p in all_probs]

        print("\n[SCIPY] Calibration Training Finished.\n")
        return df

    # -------------------------------------------------------------------------
    #  INFERENCE
    # -------------------------------------------------------------------------
    def inference(self, label_space_prob, full_vocab_prob, hidden_state):
        """
        Use the learned parameters to predict a new distribution over self.label_space.
        For n_label>2, do a softmax. For n_label=2, it effectively reduces to logistic.
        """
        if not self.fitted_:
            raise RuntimeError("lr_calib_scipy model not fitted yet.")

        # Build feature vector from label_space_prob
        x = self._make_features(label_space_prob)  # shape [n_dim]
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
    #  HELPER: Permutation
    # -------------------------------------------------------------------------
    def _permutate(self, elements, k):
        """
        Same logic from your original code to generate permutations of indexes.
        """
        if k == 0:
            return [list(perm) for perm in itertools.permutations(elements, r=1)]
        else:
            extended_permutations = [
                list(base_perm + (extra_elem,))
                for base_perm in itertools.permutations(elements, r=k)
                for extra_elem in elements if extra_elem not in base_perm
            ]
            return extended_permutations



##############
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import itertools


class lr_calib_scipy_deep(calibration):
    """
    Multiclass logistic regression calibration with invariance penalty and constraints
    """

    def __init__(
        self,
        label_space,
        use_invariance=True,
        lambda_invariance=1.0,
        invariance_loss_type='mse',
        constraint=False,
        max_iter=100,
        verbose=False,
        k=None,
        dic=None
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
        self.k = k
        self.dic = dic

        # Coefficients matrix (2 x n_features) where n_features = n_label-1
        self.coef_ = None
        self.fitted_ = False

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _negative_log_likelihood(self, params, X, Y):
        """Multiclass negative log-likelihood"""
        n_features = X.shape[1]
        b0 = params[:n_features]
        b1 = params[n_features:]
        
        logits = np.zeros((X.shape[0], self.n_label))
        logits[:, 1:] = b0 + b1 * X  # X shape (n_samples, n_features)
        
        probs = self._softmax(logits)
        eps = 1e-9
        return -np.sum(np.log(probs[np.arange(len(Y)), Y] + eps))

    def _invariance_penalty(self, params, X, pairs):
        """Multiclass invariance penalty"""
        n_features = X.shape[1]
        b0 = params[:n_features]
        b1 = params[n_features:]
        
        logits = np.zeros((X.shape[0], self.n_label))
        logits[:, 1:] = b0 + b1 * X
        probs = self._softmax(logits)
        
        total_pen = 0.0
        for i, j in pairs:
            p_i = probs[i]
            p_j = probs[j]
            if self.invariance_loss_type == 'mse':
                total_pen += np.sum((p_i - p_j) ** 2)
            elif self.invariance_loss_type == 'l1':
                total_pen += np.sum(np.abs(p_i - p_j))
            elif self.invariance_loss_type == 'sym_ce':
                ce_ij = -np.sum(p_j * np.log(p_i + 1e-9))
                ce_ji = -np.sum(p_i * np.log(p_j + 1e-9))
                total_pen += ce_ij + ce_ji
            else:
                total_pen += np.sum((p_i - p_j) ** 2)
        return total_pen

    def _objective(self, params, X, Y, pairs):
        return self._negative_log_likelihood(params, X, Y) + \
               self.lambda_invariance * self._invariance_penalty(params, X, pairs)

    def train(
        self,
        default_prompt_maker: callable,
        feedforward: callable,
        demonstration_set=None,
        k=4,
        demonstration_set_index=None
    ):
        """Training procedure with constraint handling"""
        # 1. Build training data
        train_indexes = self._permutate(demonstration_set_index, k)
        
        probs = []
        labels = []
        queries = []
        
        total = len(train_indexes)
        for i, ind in enumerate(train_indexes):
            demonstration_samples = ind[:k]
            query_sample = ind[k]
            query = demonstration_set[query_sample][0]
            label = demonstration_set.get_label(query_sample)
            prompt = default_prompt_maker(
                [demonstration_set[demonstration_samples[j]] for j in range(k)],
                query
            )
            label_space_probs = feedforward(prompt=prompt, label_space=self.label_space)
            probs.append(label_space_probs)
            labels.append(label)
            queries.append(query_sample)

        # 2. Feature engineering
        X_list = []
        y_list = []
        for pr, lab in zip(probs, labels):
            if self.n_label == 2:
                feature_val = [np.log(pr[1]/pr[0])]
            else:
                feature_val = [np.log(pr[k]/pr[0]) for k in range(1, self.n_label)]
            X_list.append(feature_val)
            y_list.append(self.label_space.index(lab))

        df = pd.DataFrame({
            "label": y_list,
            "query_index": queries,
            "features": X_list
        })

        # 3. Binary-specific lambda adjustment
        if self.n_label == 2:
            max_0 = df.loc[df['label']==0, 'features'].apply(lambda x: x[0]).max()
            min_1 = df.loc[df['label']==1, 'features'].apply(lambda x: x[0]).min()
            if (min_1 - max_0) < 0:
                self.lambda_invariance = 0.4

        # 4. Build invariance pairs
        query_map = defaultdict(list)
        for i, qid in enumerate(df["query_index"]):
            query_map[qid].append(i)
        pairs = []
        for qid, idxs in query_map.items():
            if len(idxs) < 2:
                continue
            for i1 in range(len(idxs)):
                for i2 in range(i1+1, len(idxs)):
                    pairs.append((idxs[i1], idxs[i2]))

        X = np.array(df["features"].tolist())
        Y = np.array(df["label"].values, dtype=int)

        # 5. Optimization setup
        n_features = X.shape[1]
        init_params = np.zeros(2 * n_features)
        constraints = []
        method = 'BFGS'

        # Constraint handling
        if self.constraint and self.dic is not None:
            if self.n_label == 2:
                # Binary constraint
                constraint = {
                    'type': 'ineq',
                    'fun': lambda params: self.dic[self.k] * np.abs(params[1]) - np.abs(params[0])
                }
                constraints.append(constraint)
            else:
                # Multiclass constraints
                def multiclass_constraint(params):
                    n_feat = len(params) // 2
                    constraints = []
                    for i in range(n_feat):
                        b0 = params[i]
                        b1 = params[n_feat + i]
                        constraints.append(self.dic[self.k] * np.abs(b1) - np.abs(b0))
                    return np.array(constraints)
                
                constraints.append({'type': 'ineq', 'fun': multiclass_constraint})
            
            method = 'trust-constr'

        res = minimize(
            self._objective,
            init_params,
            args=(X, Y, pairs),
            method=method,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )
        
        self.coef_ = res.x
        self.fitted_ = True

        # 6. Generate predictions
        logits = np.zeros((len(X), self.n_label))
        logits[:, 1:] = self.coef_[:n_features] + self.coef_[n_features:] * X
        pred_probs = self._softmax(logits)
        
        for i in range(self.n_label):
            df[f'score_{i}'] = pred_probs[:, i]
        df['predicted'] = np.argmax(pred_probs, axis=1)

        return df

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        """Multiclass probability prediction"""
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet")
            
        if self.n_label == 2:
            feature = [np.log(label_space_prob[1]/label_space_prob[0])]
        else:
            feature = [np.log(label_space_prob[k]/label_space_prob[0]) for k in range(1, self.n_label)]
        
        n_features = len(feature)
        logits = np.zeros(self.n_label)
        logits[1:] = self.coef_[:n_features] + self.coef_[n_features:] * feature
        return self._softmax(logits.reshape(1, -1))[0]

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)

    def _permutate(self, elements, k):
        """Permutation generator for demonstration selection"""
        if k == 0:
            return [list(perm) for perm in itertools.permutations(elements, r=1)]
        else:
            return [
                list(base_perm + (extra_elem,))
                for base_perm in itertools.permutations(elements, r=k)
                for extra_elem in elements if extra_elem not in base_perm
            ]
            
            
###########################
#multi
############################
class lr_calib_scipy_2_cos(calibration):
    """
    This class implements a calibration model that always uses class 0 as a reference.
    For each non-reference class c (c = 1, ..., n_label-1), the calibration is defined as:
    
      log(P*(y=c|x)/P*(y=0|x)) = b_c + sum_{j=1}^{n_label-1} w_{c,j} * log(P(y=j|x)/P(y=0|x))
    
    The calibrated probabilities are computed via a softmax with the reference class fixed:
    
      P*(y=0|x) = 1 / (1 + sum_{c=1}^{n_label-1} exp(b_c + <w_c, x>))
      P*(y=c|x) = exp(b_c + <w_c, x>) / (1 + sum_{k=1}^{n_label-1} exp(b_k + <w_k, x>))
    
    A constraint is imposed on the calibration parameters for non-reference classes.
    For each class c (c >= 1), let p_c = [b_c, w_{c,1}, ..., w_{c,n_label-1}],
    and define the target vector v^c = [0, ..., 0, 1, 0, ..., 0] with the 1 in the c-th
    position (of the weight part). We require that the average cosine similarity
       (1/(n_label-1)) * sum_{c=1}^{n_label-1} (p_c · v^c)/||p_c||_2
    is at least a constant cosine_threshold.
    
    In the binary case (n_label=2), this reduces to a single constraint on class 1.
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
        cosine_threshold=0.9  # minimum required average cosine similarity
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
        # We optimize only for non-reference classes.
        # For each class c=1,...,n_label-1, we have (1 + (n_label-1)) parameters.
        # Total parameter count = (n_label - 1) * n_label.
        self.params_ = None
        self.fitted_ = False

    # -------------------------------------------------------------------------
    # Feature building: Compute features as log-ratios with respect to class 0.
    # -------------------------------------------------------------------------
    def _make_features(self, prob_vector):
        eps = 1e-9
        base = prob_vector[0] + eps
        feats = []
        # Features: for j=1,...,n_label-1, x_j = log(prob[j]/prob[0])
        for i in range(1, self.n_label):
            feats.append(np.log(prob_vector[i] / base))
        return np.array(feats, dtype=float)

    # -------------------------------------------------------------------------
    # Unpack parameters: Construct a full parameter matrix with a fixed zero row for class 0.
    # The optimized parameters correspond only to classes 1,..., n_label-1.
    # Each non-reference row is of length (n_label-1 + 1) = n_label.
    # -------------------------------------------------------------------------
    def _unpack_params(self, params):
        n_dim = self.n_label - 1  # feature dimension
        # Reshape optimized parameters into (n_label-1) x (n_dim+1)
        non_ref = params.reshape(self.n_label - 1, n_dim + 1)
        # Prepend a row of zeros for the reference class (class 0)
        ref = np.zeros((1, n_dim + 1))
        param_matrix = np.vstack([ref, non_ref])
        return param_matrix

    # -------------------------------------------------------------------------
    # Compute logits: For class 0, logit is fixed to 0; for c>=1, use the calibrated transformation.
    # -------------------------------------------------------------------------
    def _compute_logits(self, param_matrix, x):
        # x is the feature vector of length n_label-1
        logits = [0.0]  # logit for class 0 is 0
        for c in range(1, self.n_label):
            b_c = param_matrix[c, 0]
            w_c = param_matrix[c, 1:]
            logits.append(b_c + np.dot(w_c, x))
        return np.array(logits)

    # -------------------------------------------------------------------------
    # Negative Log-Likelihood (using softmax over the calibrated logits)
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
    # Invariance penalty (if applicable)
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
    # Full objective: NLL + lambda_invariance * Invariance Penalty
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
                f"\rProcess: {int((i + 1) / total * 100)}% "
                f"[{'>>' * int((i + 1) / total * 32)}"
                f"{'.' * (32 - int((i + 1) / total * 32))}] "
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

        # Build feature matrix and labels.
        X_list = []
        y_list = []
        for pr, lab in zip(probs_list, labels_list):
            x_vec = self._make_features(pr)
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

        # Optional: adaptive lambda_invariance for binary case.
        if self.n_label == 2:
            zero_idx = (df["label"] == 0)
            one_idx  = (df["label"] == 1)
            if zero_idx.any() and one_idx.any():
                x_zero = [x[0] for x, lbl in zip(X, Y) if lbl == 0]
                x_one  = [x[0] for x, lbl in zip(X, Y) if lbl == 1]
                if len(x_zero) > 0 and len(x_one) > 0:
                    max_0 = max(x_zero)
                    min_1 = min(x_one)
                    if (min_1 - max_0) < 0:
                        self.lambda_invariance = 0.4

        # Prepare initial parameters.
        # Number of parameters = (n_label-1)*((n_label-1)+1) = (n_label-1)*n_label.
        n_dim = self.n_label - 1
        init_params = np.zeros((self.n_label - 1) * (n_dim + 1), dtype=float)

        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        # -------------------------------
        # BUILD CONSTRAINTS (cosine similarity constraint)
        # -------------------------------
        constraints_list = []
        if self.constraint:
            def cosine_constraint(params):
                param_matrix = self._unpack_params(params)
                cosine_sum = 0.0
                count = 0
                # Only consider non-reference classes: c = 1, ..., n_label-1.
                for c in range(1, self.n_label):
                    p_c = param_matrix[c, :]  # shape: (n_label,)
                    norm = np.linalg.norm(p_c)
                    if norm < 1e-9:
                        cosine = 0.0
                    else:
                        # Target vector for class c: zeros in bias, and 1 at index c.
                        target = np.zeros(self.n_label)
                        target[c] = 1.0
                        cosine = np.dot(p_c, target) / norm
                    cosine_sum += cosine
                    count += 1
                average_cosine = cosine_sum / count if count > 0 else 0.0
                return average_cosine - self.cosine_threshold
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
            
######
class lr_calib_scipy_2_cos_global(calibration):
    """
    This class implements a calibration model that always uses class 0 as a reference.
    For each non-reference class c (c = 1, ..., n_label-1), the calibration is defined as:
    
      log(P*(y=c|x)/P*(y=0|x)) = b_c + sum_{j=1}^{n_label-1} w_{c,j} * log(P(y=j|x)/P(y=0|x))
    
    The calibrated probabilities are computed via a softmax with the reference class fixed:
    
      P*(y=0|x) = 1 / (1 + sum_{c=1}^{n_label-1} exp(b_c + <w_c, x>))
      P*(y=c|x) = exp(b_c + <w_c, x>) / (1 + sum_{k=1}^{n_label-1} exp(b_k + <w_k, x>))
    
    In previous versions a per-class cosine constraint was imposed. In this version we instead build a global
    cosine constraint. We flatten all the parameters for non-reference classes into a single vector
    
         theta = [b_1, w_{1,1}, ..., w_{1,n_label-1}, b_2, w_{2,1}, ..., w_{2,n_label-1}, ..., b_{n_label-1}, w_{n_label-1,1}, ..., w_{n_label-1,n_label-1}],
    
    and we construct a target vector v block‐wise. For each non‐reference class c (c = 1,..., n_label-1), we define
         v^(c) = [0, ..., 0, 1, 0, ..., 0],
    a vector of length n_label (with the 1 in the c-th position, corresponding to w_{c,c}). The full target vector is
         v = [v^(1), v^(2), ..., v^(n_label-1)].
    The global cosine constraint is:
    
         (theta · v) / ||theta||_2 ≥ cosine_threshold.
    
    In the binary case (n_label = 2) this reduces to the single constraint on class 1.
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
        cosine_threshold=0.9  # minimum required global cosine similarity
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

        # We optimize only for non-reference classes.
        # For each non-reference class c=1,...,n_label-1, we have (1 + (n_label-1)) parameters.
        # Total parameter count = (n_label - 1) * n_label.
        self.params_ = None
        self.fitted_ = False

    # -------------------------------------------------------------------------
    # Feature building: Compute features as log-ratios with respect to class 0.
    # -------------------------------------------------------------------------
    def _make_features(self, prob_vector):
        eps = 1e-9
        base = prob_vector[0] + eps
        feats = []
        for i in range(1, self.n_label):
            feats.append(np.log(prob_vector[i] / base))
        return np.array(feats, dtype=float)

    # -------------------------------------------------------------------------
    # Unpack parameters: Construct a full parameter matrix with a fixed zero row for class 0.
    # Each non-reference row is of length n_label (1 bias + n_label-1 weights).
    # -------------------------------------------------------------------------
    def _unpack_params(self, params):
        n_dim = self.n_label - 1  # feature dimension
        non_ref = params.reshape(self.n_label - 1, n_dim + 1)
        ref = np.zeros((1, n_dim + 1))
        param_matrix = np.vstack([ref, non_ref])
        return param_matrix

    # -------------------------------------------------------------------------
    # Compute logits: For class 0 the logit is 0; for c>=1, use the calibrated transformation.
    # -------------------------------------------------------------------------
    def _compute_logits(self, param_matrix, x):
        logits = [0.0]  # class 0
        for c in range(1, self.n_label):
            b_c = param_matrix[c, 0]
            w_c = param_matrix[c, 1:]
            logits.append(b_c + np.dot(w_c, x))
        return np.array(logits)

    # -------------------------------------------------------------------------
    # Negative Log-Likelihood (using softmax over calibrated logits)
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
    # Invariance penalty (if applicable)
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
    # Full objective: NLL + lambda_invariance * Invariance Penalty
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
                f"\rProcess: {int((i + 1) / total * 100)}% "
                f"[{'>>' * int((i + 1) / total * 32)}"
                f"{'.' * (32 - int((i + 1) / total * 32))}] "
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

        # Build feature matrix and labels.
        X_list = []
        y_list = []
        for pr, lab in zip(probs_list, labels_list):
            x_vec = self._make_features(pr)
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

        # Optional: adaptive lambda_invariance for binary case.
        if self.n_label == 2:
            zero_idx = (df["label"] == 0)
            one_idx  = (df["label"] == 1)
            if zero_idx.any() and one_idx.any():
                x_zero = [x[0] for x, lbl in zip(X, Y) if lbl == 0]
                x_one  = [x[0] for x, lbl in zip(X, Y) if lbl == 1]
                if len(x_zero) > 0 and len(x_one) > 0:
                    max_0 = max(x_zero)
                    min_1 = min(x_one)
                    if (min_1 - max_0) < 0:
                        self.lambda_invariance = 0.4

        # Prepare initial parameters.
        # Number of parameters = (n_label-1) * (n_label) because each non-reference row is length n_label.
        n_dim = self.n_label - 1
        init_params = np.zeros((self.n_label - 1) * (n_dim + 1), dtype=float)

        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        # -------------------------------
        # BUILD GLOBAL COSINE CONSTRAINT
        # -------------------------------
        constraints_list = []
        if self.constraint:
            def cosine_constraint(params):
                # params is a flattened vector of shape ((n_label-1)*n_label,)
                # We build the target vector v block-wise.
                target_blocks = []
                for c in range(1, self.n_label):
                    block = np.zeros(self.n_label)  # each block is of length n_label (1 bias + n_label-1 weights)
                    block[c] = 1  # set the diagonal weight to 1 (i.e. w_{c,c})
                    target_blocks.append(block)
                target = np.concatenate(target_blocks)  # shape: ((n_label-1)*n_label,)
                norm = np.linalg.norm(params)
                if norm < 1e-9:
                    cosine = 0.0
                else:
                    cosine = np.dot(params, target) / norm
                return cosine - self.cosine_threshold
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










###################

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
###################################################################


class lr_calib_scipy_1d_cos_global(calibration):
    """
    This class implements independent (univariate) calibration functions for non-reference classes.
    For an original probability vector [P(y=0|x), P(y=1|x), ..., P(y=n_label-1|x)],
    we define the feature for each class i (i=1,...,n_label-1) as:
        x_i = log(P(y=i|x) / P(y=0|x))
    
    The calibration equations are:
        log(P*(y=i|x)/P*(y=0|x)) = b_i + w_i * x_i,  for i=1,...,n_label-1,
    with the reference class fixed (its logit is 0). The calibrated probabilities are computed as:
    
        P*(y=0|x) = 1 / (1 + sum_{i=1}^{n_label-1} exp(b_i + w_i*x_i))
        P*(y=i|x) = exp(b_i + w_i*x_i) / (1 + sum_{j=1}^{n_label-1} exp(b_j + w_j*x_j)).
    
    This version imposes a global cosine similarity constraint on the concatenated calibration parameters.
    Let
        theta = [b_1, w_1, b_2, w_2, ..., b_{n_label-1}, w_{n_label-1}]
    and define the target vector as
        v = [0, 1, 0, 1, ..., 0, 1].
    The constraint is:
        (theta dot v) / ||theta||_2 >= cosine_threshold.
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
        cosine_threshold=0.9  # global cosine similarity threshold
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
        
        # We optimize for non-reference classes only.
        # Each non-reference class i (i=1,...,n_label-1) has 2 parameters: [b_i, w_i].
        # Total parameters = 2*(n_label-1).
        self.params_ = None
        self.fitted_ = False

    # -------------------------------------------------------------------------
    # Feature building: For a probability vector, compute features:
    # x = [log(P(y=1)/P(y=0)), ..., log(P(y=n_label-1)/P(y=0))]
    # -------------------------------------------------------------------------
    def _make_features(self, prob_vector):
        eps = 1e-9
        base = prob_vector[0] + eps
        feats = []
        for i in range(1, self.n_label):
            feats.append(np.log(prob_vector[i] / base))
        return np.array(feats, dtype=float)  # shape: (n_label-1,)

    # -------------------------------------------------------------------------
    # Unpack parameters:
    # The optimized parameters (a 1D array of length 2*(n_label-1)) are reshaped into a matrix.
    # Row i (for i=1,...,n_label-1) contains [b_i, w_i].
    # For the reference class (class 0), we set parameters to zero.
    # -------------------------------------------------------------------------
    def _unpack_params(self, params):
        non_ref = params.reshape(self.n_label - 1, 2)  # shape: (n_label-1, 2)
        ref = np.zeros((1, 2))
        param_matrix = np.vstack([ref, non_ref])
        return param_matrix  # shape: (n_label, 2)

    # -------------------------------------------------------------------------
    # Compute logits:
    # For a sample with feature vector x (length n_label-1),
    # the logit for class 0 is fixed to 0,
    # and for class i (i>=1): logit_i = b_i + w_i * x[i-1]
    # -------------------------------------------------------------------------
    def _compute_logits(self, param_matrix, x):
        logits = [0.0]  # class 0
        for i in range(1, self.n_label):
            b_i = param_matrix[i, 0]
            w_i = param_matrix[i, 1]
            logits.append(b_i + w_i * x[i - 1])
        return np.array(logits)

    # -------------------------------------------------------------------------
    # Negative Log-Likelihood (using softmax over calibrated logits)
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

        # Build features and label arrays.
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

        # Prepare initial parameters: shape = (2*(n_label-1),)
        init_params = np.zeros((self.n_label - 1) * 2, dtype=float)

        def func_to_minimize(params):
            return self._objective(params, X, Y, pairs)

        # -------------------------------
        # BUILD GLOBAL COSINE CONSTRAINT
        # -------------------------------
        constraints_list = []
        if self.constraint:
            def cosine_constraint(params):
                # params is theta = [b_1, w_1, ..., b_{n_label-1}, w_{n_label-1}]
                # Build target vector: [0, 1, 0, 1, ..., 0, 1]
                target = np.array([0, 1] * (self.n_label - 1))
                norm = np.linalg.norm(params)
                if norm < 1e-9:
                    cosine = 0.0
                else:
                    cosine = np.dot(params, target) / norm
                return cosine - self.cosine_threshold
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
