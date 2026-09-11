import os
import random
import numpy as np
import torch
from torch import nn
# from torchmetrics.functional import pairwise_cosine_similarity
from torch.linalg import norm
torch.set_printoptions(precision=2)

class ContinuallyLearningPrototypes(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self,
                 feature_size,
                 backbone=None,
                 sim_metric='dot_product',
                 n_protos=500,
                 num_classes=2,
                 alpha_init=1,
                 sim_th_init=0.45,
                 n_wta=5,
                 k_hit=1,
                 k_miss=0.5,
                 tau_sim_th_pos=100,
                 tau_sim_th_neg=100,
                 k_sim_th_pos=0.9,
                 k_sim_th_neg=1.1,
                 device='cuda',
                 supervised=True,
                 adaptive_th=False,
                 adaptive_protos=True,
                 learn_outliers=True,
                 verbose=0,
                 learning_period=None):
        """
        Init function for the CLP model.
        :param feature_size: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(ContinuallyLearningPrototypes, self).__init__()

        # CLP parameters
        self.feature_size = feature_size
        self.sim_metric = sim_metric
        self.n_protos = n_protos
        self.num_classes = num_classes
        self.alpha_init = alpha_init
        self.sim_th_init = sim_th_init
        self.n_wta = n_wta
        self.k_hit = k_hit
        self.k_miss = k_miss
        self.tau_sim_th_pos = tau_sim_th_pos
        self.tau_sim_th_neg = tau_sim_th_neg
        self.k_sim_th_pos = k_sim_th_pos
        self.k_sim_th_neg = k_sim_th_neg
        self.device = device
        self.verbose = verbose
        self.adaptive_th = adaptive_th
        self.adaptive_protos = adaptive_protos
        self.supervised = supervised
        self.learn_outliers = learn_outliers
        self.learning_period = learning_period

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for CLP
        self.prototypes_ = torch.zeros(self.n_protos, self.feature_size).to(self.device)
        self.proto_labels_ = -1 * torch.ones((self.n_protos, 1)).long().to(self.device)
        self.sim_th_ = self.sim_th_init * torch.ones((self.n_protos, 1)).to(self.device)
        # Per-prototype scalars (learning rate, goodness) are only ever read
        # and written one slot at a time, never in a vectorised device op, so
        # they live on the host: a 0-dim CPU tensor used in a CUDA op is passed
        # as a kernel scalar (no copy, no host-device sync) and the float32
        # arithmetic is identical to the former on-device broadcast
        self.alphas_ = self.alpha_init * torch.ones((self.n_protos, 1))
        self.goodness_ = torch.ones((self.n_protos, 1))
        self.classes_ = []
        self.next_alloc_id_ = 0
        self._bind_host_state()
        self.mistaken_proto_inds_ = []

        self.n_outlier = 0
        self.n_error = 0
        self.num_updates = 0


    @torch.no_grad()
    def fit(self, x, y, i=0):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        x = x[None, :]
        x0 = x[0]  # (d,) view; row updates operate on views of this shape

        # Similarities against allocated prototypes only (unallocated slots
        # are zero vectors: sims exactly 0, always below threshold, so
        # excluding them is behavior-identical and saves (P - M) * d MACs)
        n_alloc = self._n_allocated()
        k = min(self.n_wta, n_alloc)
        if n_alloc == 0:
            # empty store: nothing can win, the sample is allocated
            sims_sorted = None
            sum_sims = 0
            n_th_passing_protos = 0
            top_sims, top_inds = [], []
            y_int = int(y.item())
        else:
            protos = self.prototypes_[:n_alloc]
            if self.sim_metric == 'euclidean':
                sims = -torch.cdist(protos, x, p=2)
            else:
                sims = torch.mm(protos, x.T)  # (n_alloc, 1)
            below_threshold = torch.gt(self.sim_th_[:n_alloc], sims)
            sims = sims.masked_fill(below_threshold, 0)
            sims_sorted, inds_sorted = torch.topk(sims, k, dim=0)
            sims_sorted = sims_sorted.view(-1)

            # Host-side scalars for the control flow. On CUDA everything is
            # gathered on the device into one small vector and moved in a
            # single transfer (the only host-device sync of the step);
            # indices and labels are exact in float32. On CPU the values are
            # read directly.
            if sims.device.type == 'cpu':
                sum_sims = sims.sum().item()
                n_th_passing_protos = n_alloc - int(below_threshold.sum().item())
                top_sims = sims_sorted.tolist()
                top_inds = inds_sorted.view(-1).tolist()
                y_int = int(y.item())
            else:
                dt = sims.dtype
                packed = torch.cat([
                    sims.sum().view(1),
                    below_threshold.sum().view(1).to(dt),
                    sims_sorted,
                    inds_sorted.view(-1).to(dt),
                    y.view(-1).to(dt),
                ]).tolist()
                sum_sims = packed[0]
                n_th_passing_protos = n_alloc - int(packed[1])
                top_sims = packed[2:2 + k]
                top_inds = [int(v) for v in packed[2 + k:2 + 2 * k]]
                y_int = int(packed[2 + 2 * k])
        labels_host = self._labels_host

        if sum_sims > 0:
            bmu_ind = top_inds[0]
        else:
            bmu_ind = -1

        if (y_int not in self.classes_) and self.supervised:
            self.classes_.append(y_int)
            if self.verbose >= 1:
                print("Novel Label!")
            self._allocate(x0, y_int)

        # Novel instance --> Allocate
        # if no winner, because all similarities are below the given threshold, then allocate
        elif bmu_ind == -1:
            if self.verbose >= 1:
                print("Novel Instance!")
                print("Label", y_int)
            self._allocate(x0, y_int)

        # Update the winner based on its inference
        elif self.supervised:
            # If CORRECT prediction
            if labels_host[bmu_ind] == y_int:
                if self.adaptive_protos:
                    error = self._calc_err(x0, bmu_ind)
                    self._positive_update(bmu_ind, self._sim_arg(sims_sorted, top_sims, 0), error)

            # if INCORRECT prediction
            else:
                self.n_error += 1
                if self.adaptive_protos:
                    n_protos_to_update = min(self.n_wta, n_th_passing_protos)
                    positive_match = False
                    for m in range(0, n_protos_to_update):
                        next_bmu_ind = top_inds[m]
                        next_sim = self._sim_arg(sims_sorted, top_sims, m)
                        error = self._calc_err(x0, next_bmu_ind)
                        if labels_host[next_bmu_ind] == y_int:
                            # Found a correct prototype - update it positively
                            self._positive_update(next_bmu_ind, next_sim, error)
                            positive_match = True
                            break  # Exit loop after finding first correct match
                        else:
                            # Found an incorrect prototype - update it negatively
                            self._negative_update(next_bmu_ind, next_sim, error)
                    # Allocate a new prototype if none of k-winners is a positive match
                    if not positive_match:
                        self.n_outlier += 1
                        if self.learn_outliers:
                            self._allocate(x0, y_int)
                else:
                    if self.learning_period is not None:
                        if self.n_error % self.learning_period == 0:
                            if self.verbose >= 1:
                                print("Mistake, allocating...")
                            self._allocate(x0, y_int)
                    else:
                        self._allocate(x0, y_int)

        else:
            if self.adaptive_protos:
                error = self._calc_err(x0, bmu_ind)
                self._positive_update(bmu_ind, self._sim_arg(sims_sorted, top_sims, 0), error)

        self.num_updates += 1

    def _sim_arg(self, sims_sorted, top_sims, m):
        """Winner similarity as passed to the update helpers. Only the
        adaptive-threshold rule reads it numerically; it then gets the
        float32 tensor element (device arithmetic, as before) rather than
        the Python float."""
        return sims_sorted[m] if self.adaptive_th else top_sims[m]

    def _bind_host_state(self):
        """Host-side mirrors of per-slot state.

        goodness_ and alphas_ are float32 CPU tensors; the numpy views share
        their storage, so scalar bookkeeping runs in numpy float32 (same IEEE
        results as the former one-element tensor ops) without tensor
        dispatch. Labels are also kept as a Python list: they are written
        only in _allocate, so the host always knows them and the fit step
        never has to read them back from the device.
        """
        self._goodness_np = self.goodness_.numpy()
        self._alphas_np = self.alphas_.numpy()
        self._labels_host = self.proto_labels_.view(-1).tolist()
        n = self.next_alloc_id_
        # True once the last slot has been allocated (next_alloc_id_ is then
        # clamped at n_protos - 1 and that slot counts as allocated)
        self._buffer_full = bool(n < self.n_protos and self._labels_host[n] >= 0)

    def _set_alpha(self, bmu_ind):
        # alpha = alpha_init / max(goodness, 1). The former tensor expression
        # `alpha_init / g` evaluated as reciprocal(g) * alpha_init in float32
        # (Tensor.__rdiv__); the same two operations in the same order keep
        # the result bitwise for any alpha_init.
        # Explicit float32 operands: numpy 1.x promotes float32-with-Python-
        # scalar to float64, torch kept everything in float32.
        g = self._goodness_np[bmu_ind, 0]
        self._alphas_np[bmu_ind, 0] = (np.float32(1) / max(g, np.float32(1))) * np.float32(self.alpha_init)

    # Row updates take a Python int slot index and operate in place on a
    # (d,) view of the prototype matrix: no gather/scatter, no index tensor,
    # no host sync. Arithmetic order matches the former gather-based code
    # (alpha * error first, then the add, then an explicit L2 renorm), so
    # results are bitwise identical.
    def _positive_update(self, bmu_ind, sim, error):
        w = self.prototypes_[bmu_ind]
        w.add_(error * float(self._alphas_np[bmu_ind, 0]))
        w.div_(torch.linalg.vector_norm(w, 2))

        # update the threshold towards max_sim-eps
        if self.adaptive_th:
            self.sim_th_[bmu_ind] = self.sim_th_[bmu_ind] + \
            (self.k_sim_th_pos*sim - self.sim_th_[bmu_ind]) / self.tau_sim_th_pos

        if self.supervised:
            self._goodness_np[bmu_ind, 0] += np.float32(self.k_hit)
        else:
            self._goodness_np[bmu_ind, 0] += np.float32(0.5*self.k_hit)

        self._set_alpha(bmu_ind)

    def _negative_update(self, bmu_ind, sim, error):
        # update the mistaken prototype
        w = self.prototypes_[bmu_ind]
        w.sub_(error * float(self._alphas_np[bmu_ind, 0]))
        w.div_(torch.linalg.vector_norm(w, 2))

        # update the threshold of this prototype
        if self.adaptive_th:
            self.sim_th_[bmu_ind] = self.sim_th_[bmu_ind] + \
            (self.k_sim_th_neg*sim - self.sim_th_[bmu_ind]) / self.tau_sim_th_neg

        self._goodness_np[bmu_ind, 0] -= np.float32(self.k_miss)
        self._set_alpha(bmu_ind)

    # def predict(self, X, return_probas=False, thresholded=False, return_sims=False, return_voting_winner=False):
    #     """
    #     Make predictions on test data X using the learned prototypes.

    #     Args:
    #         X (torch.Tensor): An N x d tensor containing N data samples.
    #         return_probas (bool): If True, returns probability distributions over classes.
    #         thresholded (bool): If True, sets similarities below each prototype's threshold to zero.
    #         return_sims (bool): If True, returns similarities for each allocated prototype instead of class scores.
    #         return_voting_winner (bool): If True, returns one-hot vectors for the majority-voted class.

    #     Returns:
    #         torch.Tensor:
    #             Depending on arguments, returns:
    #                 - Similarities (N x allocated_prototypes) if return_sims.
    #                 - One-hot vectors for majority-voted class (N x num_classes) if return_voting_winner.
    #                 - Class logits or probabilities (N x num_classes) otherwise.
    #     """
    #     X = X.to(self.device)

    #     # Calculate similarities
    #     sims = self._calc_similarities(X).detach()

    #     # Ensure sims is 2D (n_protos, n_samples)
    #     if sims.dim() == 1:
    #         sims = sims.unsqueeze(1)

    #     # If threshold is enforced, zero out similarities below each prototype's threshold
    #     if thresholded:
    #         th_mask = torch.gt(self.sim_th_.tile((1, sims.shape[1])), sims)
    #         sims[th_mask] = 0

    #     # Early return if user wants to see raw similarities
    #     if return_sims:
    #         # Only return the allocated prototypes up to 'next_alloc_id_'
    #         return sims[:self.next_alloc_id_, :].T  # Shape => (N, allocated_prototypes)

    #     # Sort similarities to get top-k prototypes for majority voting
    #     inds_sorted = torch.argsort(sims, 0, descending=True)

    #     # Gather the top 'self.n_wta' prototypes for each sample
    #     voting_protos = inds_sorted[:self.n_wta, :]
    #     voted_labels = self.proto_labels_[voting_protos]  # Shape => (n_wta, N)

    #     # Debugging: Print similarity values and corresponding labels
    #     # print("Debugging Voting Process:")
    #     # for i in range(voted_labels.shape[1]):  # Iterate over samples
    #     #     print(f"Sample {i + 1}:")
    #     #     print(f"  Similarities: {sims[voting_protos[:, i], i].cpu().numpy()}")
    #     #     print(f"  Labels: {voted_labels[:, i].cpu().numpy()}")

    #     # Weighted voting per sample
    #     final_labels = []
    #     for i in range(voted_labels.shape[1]):
    #         # Filter out prototypes labeled -1
    #         valid_indices = voted_labels[:, i] != -1
    #         valid_labels = voted_labels[:, i][valid_indices]

    #         # Handle single-sample case
    #         if sims.shape[1] == 1:
    #             valid_sims = sims[voting_protos[:, i]][valid_indices]
    #         else:
    #             valid_sims = sims[voting_protos[:, i], i][valid_indices]

    #         # If all were -1, handle as unknown
    #         if valid_labels.numel() == 0:
    #             final_labels.append(-1)
    #             continue

    #         # Perform weighted voting
    #         label_weights = {}
    #         for label, sim in zip(valid_labels.cpu().numpy(), valid_sims.cpu().numpy()):
    #             if label not in label_weights:
    #                 label_weights[label] = 0
    #             label_weights[label] += sim

    #         # Choose the label with the highest weighted score
    #         chosen_label = max(label_weights, key=label_weights.get)
    #         final_labels.append(chosen_label)

    #     # Convert majority-voted labels to a NumPy array
    #     final_labels = np.array(final_labels)  # shape => (N,)

    #     # If user requests a one-hot style winner
    #     if return_voting_winner:
    #         n_samples = X.shape[0]
    #         one_hot_scores = torch.zeros(size=(n_samples, self.num_classes))
    #         # Mark the voted label with 1 for each sample
    #         one_hot_scores[torch.arange(n_samples), torch.tensor(final_labels)] = 1
    #         return one_hot_scores

    #     # Otherwise, build class-level similarity scores
    #     n_samples = X.shape[0]
    #     scores = torch.zeros(size=(n_samples, self.num_classes))
    #     learned_classes = torch.unique(self.proto_labels_[self.proto_labels_ > -1])

    #     # Create a mask for each label and get the max similarity for that label
    #     label_masks = [(self.proto_labels_ == label) for label in range(self.n_protos)]
    #     label_masks = torch.squeeze(torch.stack(label_masks))  # shape => (n_protos, ...)

    #     # For each learned class, compute maximum similarity
    #     for c in learned_classes:
    #         class_mask = label_masks[c, :]
    #         class_sims = sims[class_mask, :]
    #         if class_sims.numel() > 0:
    #             scores[:, c] = class_sims.max(dim=0)[0]

    #     # Return logits or probabilities
    #     if not return_probas:
    #         return scores.cpu()
    #     else:
    #         return torch.softmax(scores, dim=1).cpu()
    
    @torch.no_grad()
    def predict(self, X, return_probas=False, thresholded=False, return_sims=False, return_voting_winner=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :param thresholded: True if similarity thresholds are enforced.
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        # if self.sim_metric == 'dot_product':  # normalize x, if we use dot product similarity
        #     X = X / norm(X, dim=1).unsqueeze(1)

        # Similarities against allocated prototypes only (unallocated slots
        # are zero vectors and can never contribute a class score)
        n_alloc = self._n_allocated()
        sims = self._calc_similarities(X, n_alloc)

        if thresholded:
            # broadcast compare instead of tile + boolean index_put (which
            # forces a host sync); values are identical
            sims = sims.masked_fill(torch.gt(self.sim_th_[:n_alloc], sims), 0)

        n_samples = X.shape[0]
        proto_labels_alloc = self.proto_labels_[:n_alloc]

        if return_sims:
            return sims[:self.next_alloc_id_, :].T

        if return_voting_winner:
            # Voting labels are only needed on this path; the per-sample
            # Python loop is skipped for plain score prediction
            _, inds_sorted = torch.sort(sims, 0, descending=True)
            final_labels = []
            for i in range(inds_sorted.shape[1]):  # Iterate over samples (columns)
                # All sliced prototypes are allocated (label >= 0)
                allocated_inds = inds_sorted[:, i]
                valid_allocated_inds = allocated_inds[:self.n_wta]

                # Get labels of these allocated prototypes
                sample_labels = proto_labels_alloc[valid_allocated_inds].cpu().numpy().flatten()

                if len(sample_labels) == 0:
                    # If all labels are invalid, assign label 0 as fallback
                    final_labels.append(0)
                else:
                    label_counts = np.bincount(sample_labels)  # Count occurrences
                    max_count = np.max(label_counts)  # Find max count

                    # Find all labels with the maximum count
                    candidates = np.flatnonzero(label_counts == max_count)

                    # Deterministically select the smallest label for tie-breaking
                    final_labels.append(candidates[0])

            final_labels = np.array(final_labels)
            scores = torch.zeros(size=(n_samples, self.num_classes))
            # Set the winner label to 1 for each sample
            scores[torch.arange(n_samples), torch.tensor(final_labels)] = 1
            return scores

        # Max similarity per learned class in one scatter-max over the
        # allocated slice (was: one boolean-mask gather per learned class,
        # each a host sync). Every allocated slot carries a label >= 0, so
        # the label vector is a valid scatter index; classes without a
        # prototype keep the zero initial value, as before. max is exact.
        scores = torch.zeros(size=(n_samples, self.num_classes), device=sims.device)
        if n_alloc > 0:
            scores.scatter_reduce_(
                1,
                proto_labels_alloc.view(1, -1).expand(n_samples, n_alloc),
                sims.T.to(scores.dtype),
                reduce='amax',
                include_self=False,
            )
        scores = scores.cpu()

        # return predictions or probabilities
        if not return_probas:
            return scores
        else:
            return torch.softmax(scores, dim=1)

    def _calc_err(self, x, bmu_ind):
        """Update direction for input row x (a (d,) view) and slot bmu_ind."""
        error = 0
        if self.sim_metric == 'euclidean':
            error = x - self.prototypes_[bmu_ind]

        elif self.sim_metric == 'dot_product':
            error = x
        else:
            raise NotImplementedError("Can't compute error for cosine similarity: only implemented for Euclidean and dot product similarity")

        return error

    def _n_allocated(self):
        """Number of allocated prototype slots.

        next_alloc_id_ is clamped at n_protos - 1, so once the buffer is full
        the slot AT next_alloc_id_ is itself allocated and must be counted.
        Tracked on the host (no device probe).
        """
        n = self.next_alloc_id_
        if self._buffer_full:
            n += 1
        return n

    def _calc_similarities(self, x, n_protos=None):
        """Similarities between x and the first n_protos prototypes
        (all slots when n_protos is None)."""
        similarities = 0
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        protos = self.prototypes_ if n_protos is None else self.prototypes_[:n_protos]

        if self.sim_metric == 'euclidean':
            similarities = -torch.cdist(protos, x, p=2)

        elif self.sim_metric == 'dot_product':
            similarities = torch.mm(protos, x.T)

        # elif self.sim_metric == 'cosine':
        #     similarities = pairwise_cosine_similarity(protos, x)
        return similarities

    def _allocate(self, x, y):
        """Claim the next slot for input row x (a (d,) view) with label y (int)."""
        # print("Mistake again, allocating...")
        bmu_ind = self.next_alloc_id_
        self.proto_labels_[bmu_ind] = y
        self._labels_host[bmu_ind] = y

        w = self.prototypes_[bmu_ind]
        error = x - w
        w.add_(error * float(self._alphas_np[bmu_ind, 0]))
        w.div_(torch.linalg.vector_norm(w, 2))

        self._goodness_np[bmu_ind, 0] += np.float32(1)
        # self.alphas_[[bmu_ind]] = self.alpha_init / self.hits_[[bmu_ind]]
        if bmu_ind == self.n_protos - 1:
            self._buffer_full = True
        self.next_alloc_id_ = min(self.next_alloc_id_ + 1, self.n_protos - 1)
        # print("Total number of allocated prototypes:", self.next_alloc_id_)

    @torch.no_grad()
    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            # fit one example at a time
            for x, y in zip(batch_x_feat, batch_y):
                self.fit(x[None, :], y)

    @torch.no_grad()
    def evaluate_(self, test_loader,return_probas=True, thresholded=False, return_sims=False):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)

        if return_sims:
            probabilities = torch.empty((num_samples, self.next_alloc_id_))
        else: 
            probabilities = torch.empty((num_samples, self.num_classes))

        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, 
                                  return_probas=return_probas, 
                                  thresholded=thresholded,
                                  return_sims=return_sims)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = test_y.squeeze()
            start = end
        return probabilities, labels

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()

        d['prototypes_'] = self.prototypes_.cpu()
        d['proto_labels_'] = self.proto_labels_.cpu()
        d['alphas_'] = self.alphas_.cpu()
        d['sim_th_'] = self.sim_th_.cpu()
        d['goodness_'] = self.goodness_.cpu()
        d['classes_'] = self.classes_
        d['next_alloc_id_'] = self.next_alloc_id_

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        print('\nloading ckpt from: %s' % save_file)
        d = torch.load(os.path.join(save_file))
        self.prototypes_ = d['prototypes_'].to(self.device)
        self.proto_labels_ = d['proto_labels_'].to(self.device)
        self.alphas_ = d['alphas_'].cpu()
        self.sim_th_ = d['sim_th_'].to(self.device)
        self.goodness_ = d['goodness_'].cpu()
        self.classes_ = d['classes_']
        self.next_alloc_id_ = d['next_alloc_id_']
        self._bind_host_state()

    # Locate the best matching unit
    def _get_best_matching_unit(self, x):

        similarities = self._calc_similarities(x)
        similarities[self.mistaken_proto_inds_] -= 10000
        sims = similarities.clone().detach()

        th_passing_check = torch.gt(sims, self.sim_th_.tile((1, sims.shape[1])))
        sims_sorted, inds_sorted = torch.sort(sims, 0, descending=True)
        th_passing_sorted = torch.gather(th_passing_check, 0, inds_sorted)

        bmu_inds = torch.zeros(size=(1, sims.shape[1]))
        max_sims = torch.zeros(size=(1, sims.shape[1]))

        for i in range(sims.shape[1]):
            top_th_passing_inds = inds_sorted[th_passing_sorted[:, i], i]
            max_th_passing_sims = sims_sorted[th_passing_sorted[:, i], i]
            if len(top_th_passing_inds) > 0:
                bmu_inds[0, i] = top_th_passing_inds[0]
                max_sims[0, i] = max_th_passing_sims[0]
            else:
                bmu_inds[0, i] = -1
                max_sims[0, i] = 0

        inds_sorted = inds_sorted.long()
        top_sims, top_inds = sims_sorted[:5, :], inds_sorted[:5, :]
        if self.verbose == 2:
            for i in range(0, top_sims.shape[1], 3):
                print("-----------------------------------------------------------")
                print("sims:  ", top_sims[:, i].t().data)
                print("simth: ", self.sim_th_[top_inds[:, i]].t().data)
                print("labels:", self.proto_labels_[top_inds[:, i]].t().data)
                print("alphas:", self.alphas_[top_inds[:, i]].t().data)

        bmu_inds = bmu_inds.squeeze()
        max_sims = max_sims.squeeze()

        return bmu_inds.long(), max_sims
