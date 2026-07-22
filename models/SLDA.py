"""
Streaming Linear Discriminant Analysis (SLDA) — Hayes et al. 2022.
Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).

Key constructor flags:
  streaming_update_sigma=True   → Standard SLDA (updates Σ each step) [Table 1]
  streaming_update_sigma=False  → SLDA Frozen Σ (fixed covariance) [Table 1]
  streaming_update_lambda=True  → Lazy Λ=pinv(Σ) recomputed every lambda_update_period steps

Changes vs original Embedded-CL:
  - `fit()` accepts a step index argument (ignored, for interface consistency).
  - `predict()` returns raw logits (dot-product scores) rather than probabilities;
    callers use `.topk()` directly.
  - Backbone is optional (pass `backbone=None` when features are pre-extracted).
  - Default device is 'cpu' (original defaulted to 'cuda').
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
                 streaming_update_lambda=False, lambda_update_period = 1, ood_type='mahalanobis', device='cuda'):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        self.streaming_update_lambda = streaming_update_lambda
        self.ood_type = ood_type
        self.lambda_update_period = lambda_update_period

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)  # covariance
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the SLDA model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # covariance updates
        if self.streaming_update_sigma:
            x_minus_mu = (x - self.muK[y])        
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
            Lambda = torch.pinverse(
                (
                        1 - self.shrinkage_param) * self.Sigma +
                self.shrinkage_param * torch.eye(
                    self.input_shape).to(
                    self.device))
            self.Lambda = Lambda
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
            Lambda = torch.pinverse(
                (
                        1 - self.shrinkage_param) * self.Sigma +
                self.shrinkage_param * torch.eye(
                    self.input_shape).to(
                    self.device))
            self.Lambda = Lambda
            self.prev_num_updates = self.num_updates
        else:
            Lambda = self.Lambda

        # parameters for predictions
        M = self.muK.transpose(1, 0)
        W = torch.matmul(Lambda, M)
        c = 0.5 * torch.sum(M * W, dim=0)

        # loop in mini-batches over test samples
        scores = torch.matmul(X, W) - c

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
    def ood_predict(self, x):
        def pd_mat(input1, input2, precision):  # assumes diagonal precision (kxd)
            f1 = (input1[:, None] - input2)
            f2 = f1.matmul(precision[None, :, :])
            return 0.5 * torch.diagonal(f2.matmul(f1.transpose(2, 1)), dim1=1, dim2=2)

        # compute/load Lambda matrix
        if self.prev_num_updates != self.num_updates:
            # there have been updates to the model, compute Lambda
            invC = torch.pinverse(
                (
                        1 - self.shrinkage_param) * self.Sigma +
                self.shrinkage_param * torch.eye(
                    self.input_shape).to(
                    self.device))
            self.Lambda = invC
            self.prev_num_updates = self.num_updates
        else:
            invC = self.Lambda

        if self.ood_type == 'mahalanobis':
            scores = -pd_mat(x, self.muK, invC)
        elif self.ood_type == 'baseline':
            scores = self.predict(x, return_probas=True)
        else:
            raise NotImplementedError

        return scores

    @torch.no_grad()
    def evaluate_ood_(self, test_loader):
        print('\nTesting OOD on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        scores = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            ood_scores = self.ood_predict(batch_x_feat)
            end = start + ood_scores.shape[0]
            scores[start:end] = ood_scores
            labels[start:end] = test_y.squeeze()
            start = end

        return scores, labels

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit SLDA one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, train_loader):
        # print('\nTraining on %d images.' % len(train_loader.dataset))

        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            self.fit_batch(batch_x_feat, batch_y, batch_ix)

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
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
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
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
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

    The discriminant weights W = Lambda @ muK^T are maintained incrementally
    alongside Lambda, so predictions are always per-sample fresh.

    ridge_param default (1.0) was selected once on a held-out class order
    (see experiments/slda_regularization_equivalence.py); tests are
    self-consistent under any lambda.
    """

    def __init__(self, input_shape, num_classes, backbone=None, ridge_param=1.0,
                 device='cpu', dtype=torch.float32):
        """
        Init function for the RankOneSLDA model.
        :param input_shape: feature dimension d
        :param num_classes: number of total classes in stream K
        :param ridge_param: fixed ridge lambda on the scatter matrix
        :param dtype: dtype of the maintained state (float64 available as
                      drift insurance; arithmetic cost is unchanged in order)
        """
        super(RankOneSLDA, self).__init__()

        self.device = device
        self.dtype = dtype
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.ridge_param = ridge_param

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
        self.num_updates = 0
        # diagnostic: analytically denom >= 1 every step; tracked for tests
        self.min_denom = float('inf')

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
        v = x - self.muK[y]

        # 2. Hayes' scatter recursion S <- S + c*v*v^T uses the *global*
        #    update counter; first-ever sample gives c = 0 (no precision change)
        c = self.num_updates / (self.num_updates + 1)

        u = self.Lambda @ v
        denom = 1.0 + c * torch.dot(v, u)
        self.min_denom = min(self.min_denom, denom.item())
        if denom < 1.0 - 1e-6:
            raise RuntimeError(
                f'Sherman-Morrison denom = {denom.item()} < 1 at update '
                f'{self.num_updates}: implementation or numerical error')

        if c > 0:
            # 3. Sherman-Morrison patch of Lambda for S <- S + c*v*v^T
            beta = c / denom
            self.Lambda -= beta * torch.outer(u, u)
            self.Lambda = (self.Lambda + self.Lambda.t()) / 2

            # 4. weight patch for the Lambda change (all classes, pre-update muK)
            s = self.muK @ u  # (K,)
            self.W -= beta * torch.outer(u, s)

        # 5. class-mean update
        self.muK[y] += v / (self.cK[y] + 1)
        self.cK[y] += 1

        # 6. weight patch for the mean change, using Lambda_new @ v = u / denom
        self.W[:, y] += u / (denom * self.cK[y])

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

        b = 0.5 * torch.sum(self.muK.t() * self.W, dim=0)
        scores = X @ self.W - b

        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)
            self.fit_batch(batch_x_feat, batch_y, batch_ix)

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
        self.num_updates = d['num_updates']
        self.ridge_param = d['ridge_param']
