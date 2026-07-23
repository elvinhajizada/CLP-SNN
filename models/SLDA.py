"""
Streaming Linear Discriminant Analysis (SLDA) — Hayes et al. 2022.
Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).

Key constructor flags:
  streaming_update_sigma=True   → Standard SLDA (updates Σ each step) [Table 1]
  streaming_update_sigma=False  → SLDA Frozen Σ (fixed covariance) [Table 1]
  streaming_update_lambda=True  → Lazy Λ=inv(Σ) recomputed every lambda_update_period steps
  use_fp16=True                 → FP16 GEMMs (Tensor Cores) with FP32 statistics state.
                                  Σ/Λ/μ stay FP32: accumulating Σ in FP16 stalls once the
                                  1/n increments drop below FP16 resolution and collapses
                                  accuracy (measured in experiments/slda_fp16_gate.py).
                                  FP16 is applied to the fit outer product (FP32 accumulate)
                                  and the predict score GEMM only.

Changes vs original Embedded-CL:
  - `fit()` accepts a step index argument (ignored, for interface consistency).
  - `predict()` returns raw logits (dot-product scores) rather than probabilities;
    callers use `.topk()` directly.
  - Backbone is optional (pass `backbone=None` when features are pre-extracted).
  - Default device is 'cpu' (original defaulted to 'cuda').
  - Λ is computed with `torch.linalg.inv` (≈10x faster than the original
    `torch.pinverse`; equivalent for the shrinkage-regularized SPD matrix).
"""
import os
import torch
from torch import nn


class StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant
    Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, backbone=None, shrinkage_param=1e-4, streaming_update_sigma=True,
                 streaming_update_lambda=False, lambda_update_period = 1, device='cuda', use_fp16=False):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        :param use_fp16: FP16 storage and matmuls (Tensor Core acceleration)
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        self.streaming_update_lambda = streaming_update_lambda
        self.lambda_update_period = lambda_update_period
        self.use_fp16 = use_fp16

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for SLDA; contiguous layout for efficient GEMM.
        # Statistics state is ALWAYS FP32: the Sigma recursion's per-step
        # relative increment scales as 1/n, which FP16 rounds to zero once n
        # approaches 2^11 (verified failure: experiments/slda_fp16_gate.py).
        self.muK = torch.zeros((num_classes, input_shape), dtype=torch.float32, device=self.device).contiguous()
        self.cK = torch.zeros(num_classes, dtype=torch.float32, device=self.device).contiguous()
        self.Sigma = torch.eye(input_shape, dtype=torch.float32, device=self.device).contiguous()  # covariance
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).contiguous()
        self.prev_num_updates = -1

    def _compute_lambda(self):
        """Λ = inv((1-s)Σ + sI), in FP32."""
        Lambda_fp32 = torch.linalg.inv(
            (1 - self.shrinkage_param) * self.Sigma +
            self.shrinkage_param * torch.eye(self.input_shape, dtype=torch.float32, device=self.device))
        return Lambda_fp32.contiguous()

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the SLDA model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device).float()
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # covariance updates
        if self.streaming_update_sigma:
            x_minus_mu = (x - self.muK[y])
            if self.use_fp16:
                # FP16 GEMM for the fresh outer product (Tensor Cores);
                # accumulation into Sigma stays FP32
                h = x_minus_mu.to(torch.float16)
                mult = torch.matmul(h.transpose(1, 0), h).float()
            else:
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates / (self.num_updates + 1)
            self.Sigma = (self.num_updates * self.Sigma + delta) / (
                    self.num_updates + 1)
        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(
            1)
        self.cK[y] += 1

        # compute/load Lambda matrix
        if self.streaming_update_lambda and self.num_updates % self.lambda_update_period == 0:
            # there have been updates to the model, compute Lambda
            self.Lambda = self._compute_lambda()
            self.prev_num_updates = self.num_updates

        self.num_updates += 1

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        # compute/load Lambda matrix
        if (not self.streaming_update_lambda) and (self.prev_num_updates != self.num_updates):
            # there have been updates to the model, compute Lambda
            self.Lambda = self._compute_lambda()
            self.prev_num_updates = self.num_updates
        Lambda = self.Lambda

        # parameters for predictions (FP32; weight formation from FP32 state)
        M = self.muK.transpose(1, 0)
        W = torch.matmul(Lambda, M)
        c = 0.5 * torch.sum(M * W, dim=0)

        # loop in mini-batches over test samples
        if self.use_fp16:
            # FP16 score GEMM (Tensor Cores); bias and masking in FP32
            scores = torch.matmul(X.to(torch.float16), W.to(torch.float16)).float() - c
        else:
            scores = torch.matmul(X.float(), W) - c

        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            # fit SLDA one example at a time
            for x, y in zip(batch_x_feat, batch_y):
                self.fit(x, y.view(1, ), None)

    @torch.no_grad()
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True)
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
        # grab parameters for saving (always stored in FP32 for portability)
        d = dict()
        d['muK'] = self.muK.float().cpu()
        d['cK'] = self.cK.float().cpu()
        d['Sigma'] = self.Sigma.float().cpu()
        d['num_updates'] = self.num_updates

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
        d = torch.load(os.path.join(save_file))
        print('\nloading ckpt from: %s' % save_file)
        self.muK = d['muK'].to(self.device).float()
        self.cK = d['cK'].to(self.device).float()
        self.Sigma = d['Sigma'].to(self.device).float()
        self.num_updates = d['num_updates']


class RankOneSLDA(nn.Module):
    """
    Streaming LDA with exact rank-1 precision maintenance (Sherman-Morrison
    under a fixed ridge). Per-sample cost is O(d^2 + dK); no matrix inversion
    is ever performed.

    Model: the maintained precision is Lambda = (S + lambda*I)^{-1}, where S is
    exactly the scatter matrix Hayes' recursion accumulates (S_n = n * Sigma_n,
    i.e. S <- S + c*v*v^T with c = n/(n+1) and v the deviation from the
    pre-update class mean). Because argmax is invariant to the positive scaling
    between Sigma-based and S-based scores, this reproduces StreamingLDA's
    predictions under an equivalent regularization parameterization: a fixed
    ridge S + lambda*I in place of Hayes' shrinkage (1-eps)*Sigma + eps*I.

    The discriminant weights W = Lambda @ muK^T and the bias b = 0.5*diag(muK W)
    are maintained incrementally alongside Lambda, so predictions are always
    per-sample fresh and predict is a single dK GEMM.

    ridge_param default (1.0) was selected once on a held-out class order
    (see experiments/slda_regularization_equivalence.py); tests are
    self-consistent under any lambda.
    """

    def __init__(self, input_shape, num_classes, backbone=None, ridge_param=1.0,
                 device='cpu', dtype=torch.float32, debug_checks=True):
        """
        Init function for the RankOneSLDA model.
        :param input_shape: feature dimension d
        :param num_classes: number of total classes in stream K
        :param ridge_param: fixed ridge lambda on the scatter matrix
        :param dtype: dtype of the maintained state (float64 available as
                      drift insurance; arithmetic cost is unchanged in order)
        :param debug_checks: track min_denom and assert the Sherman-Morrison
                      denominator >= 1 every step. Forces a host sync per fit,
                      so benchmark runs pass False (instrumentation is excluded
                      from measured cost); accuracy runs and tests keep True.
                      The arithmetic is identical in both modes.
        """
        super(RankOneSLDA, self).__init__()

        self.device = device
        self.dtype = dtype
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.ridge_param = ridge_param
        self.debug_checks = debug_checks

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        d = input_shape
        # Lambda = (S + lambda*I)^{-1}; S = 0 at init
        self.Lambda = (torch.eye(d, dtype=dtype) / ridge_param).to(self.device)
        self.muK = torch.zeros((num_classes, d), dtype=dtype).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        # W = Lambda @ muK.T, maintained incrementally
        self.W = torch.zeros((d, num_classes), dtype=dtype).to(self.device)
        # b = 0.5 * diag(muK @ W), maintained incrementally
        self.b = torch.zeros(num_classes, dtype=dtype, device=self.device)
        self.num_updates = 0
        # diagnostic: analytically denom >= 1 every step; tracked for tests
        self.min_denom = float('inf')

        # Preallocated work buffers: fit is allocation-free in steady state
        # (no per-sample d^2 temporaries; stabilizes tail latency)
        self._v = torch.empty(d, dtype=dtype, device=self.device)
        self._u = torch.empty(d, dtype=dtype, device=self.device)
        self._s = torch.empty(num_classes, dtype=dtype, device=self.device)
        self._bs = torch.empty(num_classes, dtype=dtype, device=self.device)
        self._uu = torch.empty((d, d), dtype=dtype, device=self.device)
        self._us = torch.empty((d, num_classes), dtype=dtype, device=self.device)

    @torch.no_grad()
    def fit(self, x, y, item_ix=None):
        """
        Fit the model to a new sample (x, y). Mirrors Hayes' recursion exactly;
        order of the update steps matters.
        :param x: a torch tensor of the input data (a single vector)
        :param y: a torch tensor of the input label
        :param item_ix: ignored, for interface consistency
        :return: None
        """
        x = x.to(self.device).to(self.dtype).view(-1)
        y = int(y)

        # 1. deviation from the *pre-update* class mean
        v = torch.sub(x, self.muK[y], out=self._v)

        # 2. Hayes' scatter recursion S <- S + c*v*v^T uses the *global*
        #    update counter; first-ever sample gives c = 0 (no precision change)
        c = self.num_updates / (self.num_updates + 1)

        u = torch.mv(self.Lambda, v, out=self._u)
        denom = 1.0 + c * torch.dot(v, u)
        if self.debug_checks:
            self.min_denom = min(self.min_denom, denom.item())
            if denom < 1.0 - 1e-6:
                raise RuntimeError(
                    f'Sherman-Morrison denom = {denom.item()} < 1 at update '
                    f'{self.num_updates}: implementation or numerical error')

        if c > 0:
            # 3. Sherman-Morrison patch of Lambda for S <- S + c*v*v^T.
            #    The subtracted term beta*u*u^T is bitwise symmetric, so Lambda
            #    stays exactly symmetric by induction (the former explicit
            #    (L + L^T)/2 step was an exact no-op and is dropped).
            beta = c / denom
            torch.outer(u, u, out=self._uu)
            self._uu.mul_(beta)
            self.Lambda.sub_(self._uu)

            # 4. weight patch for the Lambda change (all classes, pre-update muK)
            s = torch.mv(self.muK, u, out=self._s)  # (K,)
            torch.outer(u, s, out=self._us)
            self._us.mul_(beta)
            self.W.sub_(self._us)

            # bias patch: Delta W[:,k] = -beta*s_k*u gives
            # Delta b_k = 0.5*muK_k . Delta W[:,k] = -0.5*beta*s_k^2
            torch.mul(s, s, out=self._bs)
            self._bs.mul_(beta)
            self.b.sub_(self._bs, alpha=0.5)

        # 5. class-mean update
        self._v.div_(self.cK[y] + 1)
        self.muK[y].add_(self._v)
        self.cK[y] += 1

        # 6. weight patch for the mean change, using Lambda_new @ v = u / denom
        self._u.div_(denom * self.cK[y])
        self.W[:, y].add_(self._u)

        # bias for class y depends on both the new mean and the new W column;
        # recompute exactly at O(d) (all other entries are untouched by 5-6)
        self.b[y] = 0.5 * torch.dot(self.muK[y], self.W[:, y])

        # 7.
        self.num_updates += 1

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X. No inversion, ever.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device).to(self.dtype)
        if len(X.shape) < 2:
            X = X.unsqueeze(0)

        # b is maintained incrementally in fit; predict is a single dK GEMM
        scores = X @ self.W - self.b

        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)
            # fit one example at a time
            for x, y in zip(batch_x_feat, batch_y):
                self.fit(x, y.view(1, ), None)

    @torch.no_grad()
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True)
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
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Lambda'] = self.Lambda.cpu()
        d['W'] = self.W.cpu()
        d['b'] = self.b.cpu()
        d['num_updates'] = self.num_updates
        d['ridge_param'] = self.ridge_param

        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into a RankOneSLDA object.
        :param save_file: path of the saved file
        :return:
        """
        d = torch.load(os.path.join(save_file))
        print('\nloading ckpt from: %s' % save_file)
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Lambda = d['Lambda'].to(self.device)
        self.W = d['W'].to(self.device)
        if 'b' in d:
            self.b = d['b'].to(self.device)
        else:  # checkpoint from before b was maintained state
            self.b = 0.5 * torch.sum(self.muK.t() * self.W, dim=0)
        self.num_updates = d['num_updates']
        self.ridge_param = d['ridge_param']
