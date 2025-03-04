import time
from typing import Literal, List, Dict, Tuple, Optional, Callable, Union
from pydantic import BaseModel, ConfigDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline
from huggingface_hub.repocard_data import EvalResult
from libs.metrics import MetricImageTextSimilarity


class EvaluatorTextToImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline_original: StableDiffusionPipeline
    pipeline_learned: StableDiffusionPipeline
    pipeline_unlearned: StableDiffusionPipeline
    prompts_forget: List[str]
    prompts_retain: List[str]
    metric_clip: MetricImageTextSimilarity

    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        eval_results = []
        images = {}

        metric_common_attributes = {
            "dataset_type": "inline-prompts",
            "task_type": "text-to-image",
        }

        for scope, prompts in {'forget': self.prompts_forget, 'retain': self.prompts_retain}.items():
            metric_common_attributes["dataset_name"] = scope.capitalize() + " set"
            scores_original: List[float] = []
            scores_learned: List[float] = []
            scores_unlearned: List[float] = []
            scores_difference_learned_unlearned: List[float] = []
            scores_difference_original_unlearned: List[float] = []
            latencies: List[float] = []

            for prompt in prompts:
                t0 = time.time()
                image_original = self.pipeline_original(prompt).images[0]  # type: ignore
                image_learned = self.pipeline_learned(prompt).images[0]  # type: ignore
                image_unlearned = self.pipeline_unlearned(prompt).images[0]  # type: ignore
                latencies.append((time.time() - t0) / 3)

                score_original = self.metric_clip.score(image_original, prompt)['clip']
                score_learned = self.metric_clip.score(image_learned, prompt)['clip']
                score_unlearned = self.metric_clip.score(image_unlearned, prompt)['clip']
                scores_original.append(score_original)
                scores_learned.append(score_learned)
                scores_unlearned.append(score_unlearned)
                scores_difference_learned_unlearned.append(score_learned - score_unlearned)
                scores_difference_original_unlearned.append(score_original - score_unlearned)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_original)
                axes[0].set_title(f"Original\nClip Score={score_original:.2f}")
                axes[0].axis("off")
                axes[1].imshow(image_learned)
                axes[1].set_title(f"Learned\nClip Score={score_learned:.2f}")
                axes[1].axis("off")
                axes[2].imshow(image_unlearned)
                axes[2].set_title(f"Unlearned\nClip Score={score_unlearned:.2f}")
                axes[2].axis("off")
                fig.suptitle(prompt, fontsize=16)
                fig.canvas.draw()
                images[prompt] = Image.fromarray(np.uint8(np.array(fig.canvas.buffer_rgba())))  # type: ignore
                plt.show()

            # Assemble metrics object
            # EvalResult: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repocard_data.py#L13
            # card_data_class: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repocard_data.py#L248
            # Some info about the fields:
            #   - task_type: str, https://hf.co/tasks
            #   - dataset_type: str, hub ID, as searchable in https://hf.co/datasets, or at least satisfying the pattern `/^(?:[\w-]+\/)?[\w-.]+$/`
            #   - dataset_name: str, pretty name
            #   - metric_type: str, whenever possible should have these names: https://hf.co/metrics
            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of original model mean (~↑)',
                metric_value=float(np.mean(scores_original)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of original model std (~↓)',
                metric_value=float(np.std(scores_original)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of learned model mean ({"~↑" if scope == "forget" else "~"})',
                metric_value=float(np.mean(scores_learned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of learned model std (~↓)',
                metric_value=float(np.std(scores_learned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of unlearned model mean ({"↓" if scope == "forget" else "↑"})',
                metric_value=float(np.mean(scores_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of unlearned model std (~↓)',
                metric_value=float(np.std(scores_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between learned and unlearned mean ({"↑" if scope == "forget" else "↓"})',
                metric_value=float(np.mean(scores_difference_learned_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between learned and unlearned std (~↓)',
                metric_value=float(np.std(scores_difference_learned_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between original and unlearned mean ({"↑" if scope == "forget" else "↓"})',
                metric_value=float(np.mean(scores_difference_original_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between original and unlearned std (~↓)',
                metric_value=float(np.std(scores_difference_original_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

        metric_common_attributes["dataset_name"] = "Forget and Retain sets"
        eval_results.append(EvalResult(
            metric_type='runtime',
            metric_name='Inference latency seconds mean(↓)',
            metric_value=float(np.mean(latencies)),
            **metric_common_attributes,  # type: ignore
        ))

        eval_results.append(EvalResult(
            metric_type='runtime',
            metric_name='Inference latency seconds std(~↓)',
            metric_value=float(np.std(latencies)),
            **metric_common_attributes,  # type: ignore
        ))

        return eval_results, images
