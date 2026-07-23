"""
Streaming Softmax with Experience Replay — Hayes et al. 2022.
Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).

Paper config: buffer=800, replay_samples=50, lr=0.001.
Note: uses raw (un-normalized) features — normalization interferes with softmax loss.

Changes vs original Embedded-CL:
  - `fit()` accepts a step index argument (ignored, for interface consistency).
  - `predict()` returns raw logits; callers use `.topk()` directly.
  - Backbone is optional (pass `backbone=None` when features are pre-extracted).
  - `use_replay=False` mode added: behaves as fine-tuning (no replay buffer).
  - Default device is 'cpu' (original defaulted to 'cuda').
  - Optional FP16 mode (`use_fp16=True`): classifier/backbone in half precision,
    FP16 buffer storage (50% memory), autocast forward passes. Replay sampling
    semantics (with-replacement) and buffer eviction are unchanged from the
    FP32 paper configuration.
  - Buffer stored in preallocated device tensors; replayed rows are gathered
    with a single index_select per fit instead of one host-to-device copy per
    row. Sampling and eviction semantics are unchanged.
"""
from collections import defaultdict
import torch
from torch import nn
import random
import numpy as np
import os

from utils import randint, CMA


class SoftmaxLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SoftmaxLayer, self).__init__()

        linear = torch.nn.Linear(input_size, output_size)
        linear.weight.data.normal_(mean=0.0, std=0.01)
        linear.bias.data.zero_()
        self.fc = linear

    def forward(self, x):
        out = self.fc(x)
        return out


class StreamingSoftmax(nn.Module):
    """
    This is an implementation of the Streaming Softmax algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, use_replay=False, backbone=None, device='cuda', lr=0.1,
                 weight_decay=1e-5, replay_samples=50, max_buffer_size=7300, use_fp16=False):
        """
        Init function for the Streaming Softmax model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param use_fp16: half-precision classifier/backbone/buffer (Tensor Cores)
        """

        super(StreamingSoftmax, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_replay = use_replay
        self.replay_samples = replay_samples
        self.max_buffer_size = max_buffer_size
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)
            if use_fp16:
                self.backbone = self.backbone.half()

        # model specific structures.
        # The buffer lives in preallocated device tensors: one gather per fit
        # instead of per-row host-to-device copies. rehearsal_ixs and
        # class_id_to_item_ix_dict keep the exact original sampling/eviction
        # bookkeeping (item_ix keys); item_ix_to_row maps each stored item to
        # its buffer row.
        self.buf_x = torch.empty((max_buffer_size, input_shape),
                                 dtype=self.dtype, device=device)
        self.buf_y = torch.empty(max_buffer_size, dtype=torch.long, device=device)
        self.item_ix_to_row = {}
        self.free_rows = []
        self._next_row = 0
        self.rehearsal_ixs = []
        self.class_id_to_item_ix_dict = defaultdict(list)
        self.num_updates = 0
        self.total_loss = CMA()
        self.msg = '\rSample %d -- train_loss=%1.6f -- buffer_size=%d'
        self.cK = torch.zeros(num_classes).to(device)

        self.classifier = SoftmaxLayer(input_shape, num_classes)
        self.classifier = self.classifier.to(device)
        if use_fp16:
            self.classifier = self.classifier.half()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def fit_fine_tune(self, x, y):
        self.classifier.train()

        # zero out grads before backward pass because they are accumulated
        self.optimizer.zero_grad()

        data_points = torch.unsqueeze(x, 0).to(self.device).to(self.dtype)
        data_labels = y.to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
            output = self.classifier(data_points)
        loss = self.criterion(output.float(), data_labels)
        loss.backward()
        self.optimizer.step()

        self.total_loss.update(loss.item())

        # print(self.msg % (self.num_updates, self.total_loss.avg, len(self.rehearsal_ixs)), end="")
        self.cK[y] += 1
        self.num_updates += 1

    def fit_replay(self, x, y, item_ix):
        self.classifier.train()

        # zero out grads before backward pass because they are accumulated
        self.optimizer.zero_grad()
        num_samples_in_buffer = len(self.rehearsal_ixs)

        if num_samples_in_buffer == 0:
            data_points = torch.unsqueeze(x, 0).to(self.device).to(self.dtype)
            data_labels = y.to(self.device)
        else:
            # buffer smaller than replay_samples: replay everything, in
            # insertion order; otherwise sample with replacement (same
            # randint stream as the original implementation)
            if num_samples_in_buffer < self.replay_samples:
                ixs = self.rehearsal_ixs
            else:
                pos = randint(len(self.rehearsal_ixs), self.replay_samples)
                ixs = [self.rehearsal_ixs[_curr_ix] for _curr_ix in pos]
            rows = torch.tensor([self.item_ix_to_row[v] for v in ixs],
                                dtype=torch.long, device=self.device)
            num_samples = len(ixs)

            data_points = torch.empty((num_samples + 1, self.input_shape),
                                      dtype=self.dtype, device=self.device)
            data_labels = torch.empty((num_samples + 1), dtype=torch.long,
                                      device=self.device)
            data_points[0] = x.to(self.device)
            data_labels[0] = y.to(self.device)
            # single device-side gather instead of one H2D copy per row
            torch.index_select(self.buf_x, 0, rows, out=data_points[1:])
            torch.index_select(self.buf_y, 0, rows, out=data_labels[1:])

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
            output = self.classifier(data_points)
        loss = self.criterion(output.float(), data_labels)
        loss.backward()
        self.optimizer.step()

        self.total_loss.update(loss.item())

        # print(self.msg % (self.num_updates, self.total_loss.avg, len(self.rehearsal_ixs)), end="")
        self.num_updates += 1
        self.cK[y] += 1

        item_key = int(item_ix) if item_ix is not None else self.num_updates
        label_int = int(y.item())

        # add new instance to buffer (stored on device in self.dtype)
        if self.free_rows:
            row = self.free_rows.pop()
        else:
            row = self._next_row
            self._next_row += 1
        self.buf_x[row] = x.to(self.device).to(self.dtype).view(-1)
        self.buf_y[row] = label_int
        self.item_ix_to_row[item_key] = row
        self.rehearsal_ixs.append(item_key)
        self.class_id_to_item_ix_dict[label_int].append(item_key)

        # if buffer is full, randomly replace previous example from class with most samples
        if len(self.rehearsal_ixs) >= self.max_buffer_size:
            # class with most samples and random item_ix from it
            max_key = max(self.class_id_to_item_ix_dict, key=lambda x: len(self.class_id_to_item_ix_dict[x]))
            max_class_list = self.class_id_to_item_ix_dict[max_key]
            rand_item_ix = random.choice(max_class_list)

            # remove the random_item_ix from all buffer references
            max_class_list.remove(rand_item_ix)
            self.free_rows.append(self.item_ix_to_row.pop(rand_item_ix))
            self.rehearsal_ixs.remove(rand_item_ix)

    def reset_buffer(self):
        """Empty the replay buffer (used by the benchmark harness after warmup)."""
        self.item_ix_to_row.clear()
        self.free_rows.clear()
        self._next_row = 0
        self.rehearsal_ixs.clear()
        self.class_id_to_item_ix_dict.clear()

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        self.classifier.eval()
        X = X.to(self.device).to(self.dtype)
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_fp16):
            scores = self.classifier(X)
        scores = scores.float()

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    def fit(self, x, y, item_ix):
        if self.use_replay:
            self.fit_replay(x, y, item_ix)
        else:
            self.fit_fine_tune(x, y)

    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            # fit model one example at a time
            for x, y, item_ix in zip(batch_x_feat, batch_y, batch_ix):
                self.fit(x, y.view(1, ), item_ix)

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

        state = {
            'state_dict': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        d = torch.load(os.path.join(save_file))
        print('\nloading ckpt from: %s' % save_file)
        self.classifier.load_state_dict(d['state_dict'])
        self.optimizer.load_state_dict(d['optimizer'])
