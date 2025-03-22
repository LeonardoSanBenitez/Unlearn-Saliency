from typing import List, Literal, Dict, Optional
from torchvision import transforms
from libs.metrics.base import Metric
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from libs.datasets import UnlearnDatasetSplit


class ClassificationAccuracy(Metric):
    metrics: List[Literal['Unlearn', 'Remaining', 'Testing']]
    unlearned_model: models
    loaders: Dict[str, DataLoader]

    def model_post_init(self, __context: dict = None) -> None:
        # Download the models, if required
        expected_keys = [UnlearnDatasetSplit.Train_forget, UnlearnDatasetSplit.Train_retain, UnlearnDatasetSplit.Test_retain]
        assert set(expected_keys) == set([key.value for key in list(self.loaders.keys())]), "Loaders should be filled"
        assert isinstance(self.unlearned_model, models), "Model not valid"
        pass

    def _accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
        """
        Computes the accuracy over the top-k predictions for the specified values of k.
        
        Args:
            output (torch.Tensor): Model predictions (logits or probabilities) with shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth labels with shape (batch_size,).
            topk (tuple): Tuple of integers indicating which top-k accuracies to compute.
        
        Returns:
            list: Accuracy values for each k in topk.
        """
        with torch.no_grad():
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

    def _compute_acc(self, metric, loader: DataLoader, unlearn: bool = False) -> float:
        losses = np.zeros(len(loader))
        top1 = np.zeros(len(loader))
        
        criterion = torch.nn.CrossEntropyLoss()

        # switch to evaluate mode
        self.unlearned_model.eval()
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        for i, (image, target) in enumerate(loader):

            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = self.unlearned_model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            prec1 = self._accuracy(output.data, target)[0]
            losses[i] = loss.item()
            top1[i] = prec1.item()

            print(f"Test: [{i}/{len(loader)}]\t" + 
            f"Loss {losses[i]:.4f} ({np.mean(losses):.4f})\t" +
            f"Accuracy {top1[i]:.3f} ({np.mean(top1):.3f})")

            print(f"Accuracy {np.mean(top1):.3f}")
        
        if unlearn == True:
            return (1 - np.mean(top1))
        else:
            return np.mean(top1)

    def score(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for metric in self.metrics:
            if metric == 'Unlearn':
                # TODO: change to test_forget, right now using train to be equal to SalUn
                self.scores[metric] = self._compute_acc(metric, self.loaders[UnlearnDatasetSplit.Train_forget], True)
            if metric == 'Remaining':
                self.scores[metric] = self._compute_acc(metric, self.loaders[UnlearnDatasetSplit.Train_retain])
            else:
                self.scores[metric] = self._compute_acc(metric, self.loaders[UnlearnDatasetSplit.Test_retain])
        
        assert len(self.scores) == len(self.metrics)
        
        return scores
