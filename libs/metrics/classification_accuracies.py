from typing import List, Literal, Dict, Optional
import numpy as np
import torch
from libs.metrics.base import Metric
from libs.datasets import UnlearnDatasetSplit


class ClassificationAccuracy(Metric):
    metrics: List[Literal['Unlearn', 'Unlearn_test', 'Remaining', 'Testing']]
    outputs: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]

    def model_post_init(self, __context: dict = None) -> None:
        expected_keys = [UnlearnDatasetSplit.Train_forget, UnlearnDatasetSplit.Test_forget, UnlearnDatasetSplit.Train_retain, UnlearnDatasetSplit.Test_retain]
        assert set(key.value for key in expected_keys) == set([key for key in list(self.outputs.keys())]), "Outputs should be filled"
        assert set(key.value for key in expected_keys) == set([key for key in list(self.targets.keys())]), "Targets should be filled"
        pass

    def _accuracy(self, output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
        """
        Computes the accuracy over the top-k predictions for the specified values of k.
        
        Args:
            output (torch.Tensor): Model predictions (logits or probabilities) with shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth labels with shape (batch_size,).
            topk (tuple): Tuple of integers indicating which top-k accuracies to compute.
        
        Returns:
            list: Accuracy values for each k in topk.
        """
        maxk = max(topk)  # Get the highest k value
        batch_size = target.size(0)

        # Get the top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (batch_size, maxk)
        pred = pred.t()  # Transpose to shape (maxk, batch_size)

        # Compare predictions with targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Shape (maxk, batch_size)

        # Compute accuracy for each k
        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)  # Sum up correct predictions
            acc_k = correct_k * 100.0 / batch_size  # Convert to percentage
            accuracies.append(acc_k.item())

        return accuracies

    def _compute_acc(self, output: torch.Tensor, target: torch.Tensor, unlearn: bool = False) -> float:
        
        top1 = self._accuracy(output.data, target)[0]
        
        if unlearn is True:
            return (100 - np.mean(top1))
        
        return np.mean(top1)

    def score(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for metric in self.metrics:
            if metric == 'Unlearn':
                scores[metric] = self._compute_acc(
                    self.outputs[UnlearnDatasetSplit.Train_forget.value],
                    self.targets[UnlearnDatasetSplit.Train_forget.value],
                    True)
            elif metric == 'Unlearn_test':
                scores[metric] = self._compute_acc(
                    self.outputs[UnlearnDatasetSplit.Test_forget.value],
                    self.targets[UnlearnDatasetSplit.Test_forget.value],
                    True)
            elif metric == 'Remaining':
                scores[metric] = self._compute_acc(
                    self.outputs[UnlearnDatasetSplit.Train_retain.value],
                    self.targets[UnlearnDatasetSplit.Train_retain.value])
            else:
                scores[metric] = self._compute_acc(
                    self.outputs[UnlearnDatasetSplit.Test_retain.value],
                    self.targets[UnlearnDatasetSplit.Test_retain.value])
        
        assert len(scores) == len(self.metrics)
        
        return scores
