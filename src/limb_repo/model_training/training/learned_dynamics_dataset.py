"""Dynamics Data loader."""

from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LearnedDynamicsDataset(Dataset):
    """Learned Dynamics Data Loader."""

    def __init__(self, h5_path: str):
        self.path = h5_path

        print("loading data...")
        self.data = h5py.File(self.path, "r")

        self.feature_keys = ["torque_action", "initial_state"]
        self.label_keys = ["result_qdd"]

        # Load all data into GPU memory and pre-concatenate
        self.features = torch.cat(
            [torch.from_numpy(self.data[key][:]) for key in self.feature_keys], dim=1
        )
        self.labels = torch.cat(
            [torch.from_numpy(self.data[key][:]) for key in self.label_keys], dim=1
        )

        # normalize features and labels between -1 and 1
        self.min_features = torch.min(self.features, dim=0).values
        self.max_features = torch.max(self.features, dim=0).values
        self.range_features = self.max_features - self.min_features

        self.min_labels = torch.min(self.labels, dim=0).values
        self.max_labels = torch.max(self.labels, dim=0).values
        self.range_labels = self.max_labels - self.min_labels

        self.features = self.normalize(
            self.features, self.min_features, self.max_features, self.range_features
        )
        # self.labels = self.normalize(
        #     self.labels, self.min_labels, self.max_labels, self.range_labels
        # )
        self.labels = torch.tanh(0.125 * self.labels)

    def normalize(
        self,
        data: torch.Tensor,
        min: torch.Tensor,
        max: torch.Tensor,
        range: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize each column independently between -1 and 1."""
        print("min data shape:", min)
        print("max data shape:", max)
        print("data shape:", data.shape)
        return 2 * (data - min) / range - 1

    def __len__(self):
        """Return the number of data points."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns feature and label tensors for a single data point from GPU
        memory."""
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
