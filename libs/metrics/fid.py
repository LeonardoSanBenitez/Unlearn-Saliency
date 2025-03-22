from typing import List, Literal, Dict
from PIL import Image
import torch
import torch_fidelity
from torchvision import transforms
from libs.metrics.base import Metric


class FrechetInceptionDistance(Metric):
    metrics: List[Literal['FID']]

    def __init__(self, metrics: Literal['FID']):
        # TODO: use pydantic's constructor, and initialize the models as post init
        # self.metrics = metrics
        pass

    def score(self, org_img_path: str, pred_img_path: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize for Inception compatibility
            transforms.ToTensor(),  # Convert image to tensor [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        img_real_tensor = transform(Image.open(org_img_path)).unsqueeze(0)
        img_fake_tensor = transform(Image.open(pred_img_path)).unsqueeze(0)

        fid_val = torch_fidelity.calculate_metrics(
            input1=org_img_path,
            input2=pred_img_path,
            metrics=['fid'],
            device= "cuda" if torch.cuda.is_available() else "cpu"
        )

        assert fid_val is not None
        scores[metric] = float(fid_val)

        return scores