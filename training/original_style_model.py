"""Experimental NN/RF 0.7/0.3 ensemble inspired by the original script."""

from __future__ import annotations

import numpy as np

from .features import ZgggWeatherFeaturePipeline
from .schema import LABEL_FIELDS


def standardize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("feature matrix must be finite and nonempty")
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (matrix - center) / scale, center, scale


class OriginalStyleEnsemble:
    """Four-duration multioutput approximation, not the original Keras weights."""

    def __init__(self, *, seed=42, epochs=30, trees=100, batch_size=256):
        if min(epochs, trees, batch_size) < 1:
            raise ValueError("training parameters must be positive")
        self.seed, self.epochs, self.trees, self.batch_size = seed, epochs, trees, batch_size

    def fit(self, rows):
        if not rows:
            raise ValueError("training rows are empty")
        import torch
        from sklearn.ensemble import RandomForestRegressor

        self.features = ZgggWeatherFeaturePipeline().fit(rows)
        raw = self.features.transform(rows)
        scaled, self.feature_center, self.feature_scale = standardize(raw)
        targets = np.asarray([[row["labels"][name] for name in LABEL_FIELDS] for row in rows], dtype=np.float64)
        if not np.isfinite(targets).all():
            raise ValueError("training targets must be finite")
        scaled_targets, self.target_center, self.target_scale = standardize(targets)
        self.forest = RandomForestRegressor(n_estimators=self.trees, max_depth=10,
                                            random_state=self.seed, n_jobs=4)
        self.forest.fit(raw, targets)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = torch.nn.Sequential(
            torch.nn.Linear(raw.shape[1], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, len(LABEL_FIELDS)),
        ).to(self.device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.001)
        x = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(scaled_targets, dtype=torch.float32, device=self.device)
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for batch in torch.randperm(len(rows), generator=generator).split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.huber_loss(network(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
        self.network = network.eval()
        return self

    def predict(self, rows):
        if not hasattr(self, "network"):
            raise ValueError("model must be fitted before prediction")
        if not rows:
            return []
        import torch

        raw = self.features.transform(rows)
        scaled = (raw - self.feature_center) / self.feature_scale
        with torch.inference_mode():
            neural = self.network(torch.as_tensor(scaled, dtype=torch.float32,
                                                  device=self.device)).cpu().numpy()
        neural = neural * self.target_scale + self.target_center
        forest = self.forest.predict(raw)
        combined = .7 * neural + .3 * forest
        combined[:, 0] = np.clip(combined[:, 0], -360, 1440)
        combined[:, 1] = np.clip(combined[:, 1], 0, 300)
        combined[:, 2] = np.clip(combined[:, 2], 0, 1440)
        combined[:, 3] = np.clip(combined[:, 3], 0, 300)
        return [{"record_id": row["record_id"],
                 **{name: float(combined[index, column]) for column, name in enumerate(LABEL_FIELDS)}}
                for index, row in enumerate(rows)]
