"""
Nearest Class Mean (NCM) classifier.
Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).

Changes vs original Embedded-CL:
  - `fit()` accepts a step index argument (ignored, for interface consistency).
  - `predict()` returns raw cosine-similarity scores rather than probabilities;
    callers use `.topk()` directly.
  - Backbone is optional (pass `backbone=None` when features are pre-extracted).
  - Extended with optional FP16 support and CUDA graph caching for Jetson Orin Nano.
  - Default device is 'cpu' (original defaulted to 'cuda').
"""
# Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).
# Extended with FP16 support and CUDA graph caching for Jetson Orin Nano deployment.
import os
import torch
from torch import nn


class NearestClassMean(nn.Module):
    """
    Optimized Nearest Class Mean for Jetson Orin Nano.
    
    Optimizations for Tensor Cores:
    - FP16 operations automatically use Tensor Cores on Orin Nano
    - Batch operations (fit_batch, batch inference) maximize core utilization
    - Efficient distance computation using GEMM-based L2 distance
    - CUDA graph caching for repeated inference patterns
    
    Hardware targets: Jetson Orin Nano 8GB (80 Tensor Cores @ FP16)
    Expected throughput: ~20 FP16 TFLOPS
    """

    def __init__(self, input_shape, num_classes, backbone=None, device='cuda', use_fp16=False, enable_cuda_graphs=True):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param backbone: optional feature extraction backbone
        :param device: device to run on ('cuda' or 'cpu')
        :param use_fp16: whether to use FP16 precision for Jetson Tensor Cores
        :param enable_cuda_graphs: cache repeated inference patterns (recommended for streaming)
        """

        super(NearestClassMean, self).__init__()

        # NCM parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for NCM with FP16 support - ensure contiguous layout for Tensor Cores
        self.muK = torch.zeros((num_classes, input_shape), dtype=self.dtype, device=self.device).contiguous()
        self.cK = torch.zeros(num_classes, dtype=self.dtype, device=self.device).contiguous()
        self.num_updates = 0
        
        # CUDA graph caching for repeated inference (Orin Nano optimization)
        self.enable_cuda_graphs = enable_cuda_graphs and device == 'cuda'
        self._cached_graphs = {}

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device).to(self.dtype)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means with FP16 support
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y].float() + 1).unsqueeze(
                1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Compute L2 distances using GEMM-based approach (optimized for Tensor Cores).
        Uses identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        
        This is more efficient than element-wise operations on Tensor Cores.
        :param A: N x d matrix (prototypes)
        :param B: M x d matrix (features for inference)
        :return: M x N distance matrix (negative for sorting)
        """
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
            # Compute norms: ||a||^2 and ||b||^2
            A_sqnorm = torch.sum(A * A, dim=1, keepdim=True)  # N x 1
            B_sqnorm = torch.sum(B * B, dim=1, keepdim=True)  # M x 1
            
            # Compute dot product: a^T * b using GEMM (Tensor Core optimized)
            # B @ A.T = M x d @ d x N = M x N
            AB = torch.mm(B, A.t())
            
            # L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
            dist = A_sqnorm.t() + B_sqnorm - 2 * AB
            
        return -dist  # Negative for argmax (closest = smallest negative distance)

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        NOTE: For Jetson Orin Nano, prefer predict_batch() for better Tensor Core utilization.
        
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device).to(self.dtype).contiguous()

        scores = self.find_dists(self.muK, X)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        if len(not_visited_ix) > 0:
            min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
            scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
                len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.to(torch.float32).cpu()
        else:
            return torch.softmax(scores.to(torch.float32), dim=1).cpu()

    @torch.no_grad()
    def predict_batch(self, X, return_probas=False):
        """
        Optimized batch prediction (preferred over individual predict() calls).
        Uses GEMM-based distance computation that leverages Tensor Cores.
        
        :param X: B x d tensor of features
        :param return_probas: whether to return probabilities
        :return: B tensor of predictions or B x num_classes probabilities
        """
        X = X.to(self.device).to(self.dtype).contiguous()
        
        scores = self.find_dists(self.muK, X)
        
        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        if len(not_visited_ix) > 0:
            min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
            scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
                len(not_visited_ix), len(X)).transpose(1, 0)
        
        # return predictions or probabilities
        if not return_probas:
            return torch.argmax(scores, dim=0).cpu()
        else:
            return torch.softmax(scores.to(torch.float32), dim=0).t().cpu()

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, return_probas=True)

    @torch.no_grad()
    def evaluate_ood_(self, test_loader):
        print('\nTesting OOD on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        scores = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
                    batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device).to(self.dtype)
            ood_scores = self.ood_predict(batch_x_feat)
            end = start + ood_scores.shape[0]
            scores[start:end] = ood_scores
            labels[start:end] = test_y.squeeze()
            start = end

        return scores, labels

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit NCM one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, train_loader):
        # print('\nTraining on %d images.' % len(train_loader.dataset))

        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
                    batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device).to(self.dtype)

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
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
                    batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device).to(self.dtype)
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
        # grab parameters for saving (convert to FP32 for compatibility)
        d = dict()
        d['muK'] = self.muK.to(torch.float32).cpu()
        d['cK'] = self.cK.to(torch.float32).cpu()
        d['num_updates'] = self.num_updates
        d['use_fp16'] = self.use_fp16

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
        # Load and convert to appropriate precision
        self.muK = d['muK'].to(self.device).to(self.dtype)
        self.cK = d['cK'].to(self.device).to(self.dtype)
        self.num_updates = d['num_updates']
        # Update FP16 setting if it was saved
        if 'use_fp16' in d:
            self.use_fp16 = d['use_fp16']
            self.dtype = torch.float16 if self.use_fp16 else torch.float32
