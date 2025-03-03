#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""



"""
Downgrades from the original script:
- Support for snr_gamma was dropped
- idem scale_lr
"""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List
import time

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub.repocard_data import EvalResult, ModelCardData

import datasets
from PIL import Image
import matplotlib.pyplot as plt
import random
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from typing import Literal, List, Dict, Tuple, Optional, Callable, Union

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")



logger = get_logger(__name__, log_level="INFO")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_forget_name: str = None,
    dataset_retain_name: str = None,
    repo_folder: str = None,
    eval_results: List[EvalResult] = [],  # whenever possible, should have this names: https://huggingface.co/metrics
    tags: List[str] = [],
):
    '''
    
    the resulting file looks like this: https://github.com/huggingface/hub-docs/blob/main/modelcard.md
    '''
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned for forgetting {dataset_forget_name} dataset, while retaining {dataset_retain_name}. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    model_card = populate_model_card(model_card, tags=tags)

    model_card.data = ModelCardData(
        model_name = repo_id,
        eval_results=eval_results
    )

    model_card.save(os.path.join(repo_folder, "README.md"))



from typing import Tuple
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline


############################################
# Unlearning algorithms
############################################
def unlearn_lora(model_original_id: str, model_lora_id: str, device: str) -> Tuple[StableDiffusionPipeline, StableDiffusionPipeline, StableDiffusionPipeline]:
    '''
    id can be both a local dir or a huggingface model id
    return pipeline_original, pipeline_learned, pipeline_unlearned
    '''
    pipeline_original = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)

    pipeline_learned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline_learned.load_lora_weights(model_lora_id, weight_name="pytorch_lora_weights.safetensors")

    pipeline_unlearned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline_unlearned.load_lora_weights(model_lora_id, weight_name="pytorch_lora_weights.safetensors")
    total: int = 0
    sum_before_invert: float = sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
    for name, param in pipeline_unlearned.unet.named_parameters():
        if "lora_A" in name:
            logger.debug(f"Inverting param {name}")
            param.data = -1 * param.data
            total += 1
    assert sum_before_invert == -sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
    assert total > 0
    logger.debug(f"Inverted {total} params")

    return pipeline_original, pipeline_learned, pipeline_unlearned

############################################
# Evaluation utilities
############################################
class ImageTextSimilarityJudge:
    metrics: List[Literal['clip']]
    _clip_score_fn: Optional[Callable]

    def __init__(self, metrics: List[Literal['clip']]):
        self.metrics = metrics

        # Download the models for the LPIPS metrics, if required
        if 'clip' in self.metrics:
            self._clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def evaluate(self, image: Union[Image.Image, np.ndarray], text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Preprocess
        image_np: np.ndarray
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        image_int = (image_np * 255).astype("uint8")

        # Calculate
        for metric in self.metrics:
            if metric == 'clip':
                assert self._clip_score_fn is not None
                scores[metric] = float(self._clip_score_fn(torch.from_numpy(image_int), text).detach())
        return scores




def eval_text_to_image_unlearning(
    pipeline_original: StableDiffusionPipeline,
    pipeline_learned: StableDiffusionPipeline,
    pipeline_unlearned: StableDiffusionPipeline,
    prompts_forget: List[str],
    prompts_retain: List[str],
    judge_clip: ImageTextSimilarityJudge,
) -> Tuple[List[EvalResult], Dict[str, List[Image.Image]]]:
    eval_results = []
    images = {}

    metric_common_attributes = {
        "dataset_type": "inline-prompts",
        "task_type": "text-to-image",
    }

    for scope, prompts in {'forget': prompts_forget, 'retain': prompts_retain}.items():
        metric_common_attributes["dataset_name"] = scope.capitalize() + " set"
        scores_original: List[float] = []
        scores_learned: List[float] = []
        scores_unlearned: List[float] = []
        scores_difference_learned_unlearned: List[float] = []
        scores_difference_original_unlearned: List[float] = []
        latencies: List[float] = []

        for prompt in prompts:
            t0 = time.time()
            image_original = pipeline_original(prompt).images[0]
            image_learned = pipeline_learned(prompt).images[0]
            image_unlearned = pipeline_unlearned(prompt).images[0]
            latencies.append((time.time() - t0)/3)

            score_original = judge_clip.evaluate(image_original, prompt)['clip']
            score_learned = judge_clip.evaluate(image_learned, prompt)['clip']
            score_unlearned = judge_clip.evaluate(image_unlearned, prompt)['clip']
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
            images[prompt] = Image.fromarray(np.uint8(np.array(fig.canvas.buffer_rgba())))
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
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of original model mean (~↑)',
            metric_value = float(np.mean(scores_original)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of original model std (~↓)',
            metric_value = float(np.std(scores_original)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of learned model mean ({"~↑" if scope == "forget" else "~"})',
            metric_value = float(np.mean(scores_learned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of learned model std (~↓)',
            metric_value = float(np.std(scores_learned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of unlearned model mean ({"↓" if scope == "forget" else "↑"})',
            metric_value = float(np.mean(scores_unlearned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score of unlearned model std (~↓)',
            metric_value = float(np.std(scores_unlearned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score difference between learned and unlearned ({"↑" if scope == "forget" else "↓"})',
            metric_value = float(np.mean(scores_difference_learned_unlearned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score difference between learned and unlearned std (~↓)',
            metric_value = float(np.std(scores_difference_learned_unlearned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score difference between original and unlearned ({"↑" if scope == "forget" else "↓"})',
            metric_value = float(np.mean(scores_difference_original_unlearned)),
            **metric_common_attributes,
        ))

        eval_results.append(EvalResult(
            metric_type = 'clip',
            metric_name = f'{scope.capitalize()}Set clip score difference between original and unlearned std (~↓)',
            metric_value = float(np.std(scores_difference_original_unlearned)),
            **metric_common_attributes,
        ))

    metric_common_attributes["dataset_name"] = "Forget and Retain sets"
    eval_results.append(EvalResult(
        metric_type = 'runtime',
        metric_name = 'Inference latency seconds mean(↓)',
        metric_value = float(np.mean(latencies)),
        **metric_common_attributes,
    ))

    eval_results.append(EvalResult(
        metric_type = 'runtime',
        metric_name = 'Inference latency seconds std(~↓)',
        metric_value = float(np.std(latencies)),
        **metric_common_attributes,
    ))



    return eval_results, images



def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_forget_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_retain_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_forget_config_name",
        type=str,
        default=None,
        help="The config of the Dataset for forgetting, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--dataset_retain_config_name",
        type=str,
        default=None,
        help="The config of the Dataset for retaining, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "Hamdy20002/COCO_Person": ("image", "text"),
}


def main():
    t0 = time.time()
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # TODO: how to flexibly receive these prompts?
    # I want to support 2 cases: when the prompts are given directly (aka list of strings), or when only the dateset ID is given and the prompts are in huggingface
    eval_prompts_forget = [
        "A naruto with blue eyes",
        "One naruto character with dark hair and brown eyes",
        "Naruto in a blue shirt and headband",
        "Naruto with a white hat and a red cross on his head",
        "Naruto in armor standing in front of a blue background",
        "A character from the anime naruto with yellow air and orange clothing, hand drawn",
        "A woman from the anime naruto, wearing clothing from the anime, standing in front of a traditional building",
        "An anime character in a white suit with a purple face, drawn in naruto-style",
        "Naruto",
        "Many characters from the series Naruto laughing and hugging each other as a family",
    ]
    eval_prompts_retain = [
        "A man with blue eyes",
        "One person with dark hair and brown eyes",
        "A man in a blue shirt and headband",
        "A man with a white hat and a red cross on his head",
        "A man in armor standing in front of a blue background",
        "A character with yellow air and orange clothing, hand drawn",
        "A woman in a suit and tie standing in front of a building",
        "An cartoon character in a white suit with a purple face",
        "A cartoon character",
        "Many people from the series Naruto laughing and hugging each other as a family",
    ]

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-s
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    t1 = time.time()

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading a dataset from the hub.
    # TODO
    dataset_forget = load_dataset(
        args.dataset_forget_name,
        args.dataset_forget_config_name,
        cache_dir=args.cache_dir,
        data_dir=None,
    )
    dataset_retain = load_dataset(
        args.dataset_retain_name,
        args.dataset_retain_config_name,
        cache_dir=args.cache_dir,
        data_dir=None,
    )


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset_forget["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_forget_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset_forget["train"] = dataset_forget["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset_forget = dataset_forget["train"].with_transform(preprocess_train)
        train_dataset_retain = dataset_retain["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_forget_dataloader = torch.utils.data.DataLoader(
        train_dataset_forget,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    train_retain_dataloader = torch.utils.data.DataLoader(
        train_dataset_retain,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_forget_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_forget_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_forget_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_forget_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_forget_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    t2 = time.time()
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_forget)} + {len(train_dataset_retain)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    similarities_gr: List[float] = []  # Cosine similarlities between \tilde g and g_r, one element per step update
    similarities_gf: List[float] = []  # Cosine similarlities between \tilde g and g_f, one element per step update

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss_forget = 0.0
        train_loss_retain = 0.0
        for step, batch_forget in enumerate(train_forget_dataloader):
            batch_retain = next(iter(train_retain_dataloader))
            min_length = min(len(batch_forget["pixel_values"]), len(batch_retain["pixel_values"]))
            batch_forget["pixel_values"] = batch_forget["pixel_values"][:min_length]
            batch_retain["pixel_values"] = batch_retain["pixel_values"][:min_length]
            batch_forget["input_ids"] = batch_forget["input_ids"][:min_length]
            batch_retain["input_ids"] = batch_retain["input_ids"][:min_length]
            assert batch_forget["pixel_values"].shape == batch_retain["pixel_values"].shape
            
            batch_forget["pixel_values"] = batch_forget["pixel_values"].to(accelerator.device)
            batch_retain["pixel_values"] = batch_retain["pixel_values"].to(accelerator.device)
            
            batch_forget["input_ids"] = batch_forget["input_ids"].to(accelerator.device)
            batch_retain["input_ids"] = batch_retain["input_ids"].to(accelerator.device)

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents_forget = vae.encode(batch_forget["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents_forget = latents_forget * vae.config.scaling_factor

                latents_retain = vae.encode(batch_retain["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents_retain = latents_retain * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise_forget = torch.randn_like(latents_forget)
                noise_retain = torch.randn_like(latents_retain)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise_forget += args.noise_offset * torch.randn(
                        (latents_forget.shape[0], latents_forget.shape[1], 1, 1), device=latents_forget.device
                    )
                    noise_retain += args.noise_offset * torch.randn(
                        (latents_retain.shape[0], latents_retain.shape[1], 1, 1), device=latents_retain.device
                    )

                bsz = latents_forget.shape[0]
                # Sample a random timestep for each image
                timesteps_forget = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_forget.device)
                timesteps_forget = timesteps_forget.long()
                timesteps_retain = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_retain.device)
                timesteps_retain = timesteps_retain.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents_forget = noise_scheduler.add_noise(latents_forget, noise_forget, timesteps_forget)
                noisy_latents_retain = noise_scheduler.add_noise(latents_retain, noise_forget, timesteps_forget)

                # Get the text embedding for conditioning
                encoder_hidden_states_forget = text_encoder(batch_forget["input_ids"], return_dict=False)[0]
                encoder_hidden_states_retain = text_encoder(batch_retain["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target_forget = noise_forget
                    target_retain = noise_retain
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target_forget = noise_scheduler.get_velocity(latents_forget, noise_forget, timesteps_forget)
                    target_retain = noise_scheduler.get_velocity(latents_retain, noise_retain, timesteps_retain)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred_forget = unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states_forget, return_dict=False)[0]
                model_pred_retain = unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states_retain, return_dict=False)[0]

                loss_forget = F.mse_loss(model_pred_forget.float(), target_forget.float(), reduction="mean")  # This is a Tensor of shape [], aka is a float
                loss_retain = F.mse_loss(model_pred_retain.float(), target_retain.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                train_loss_forget += accelerator.gather(loss_forget.repeat(args.train_batch_size)).mean().item() / args.gradient_accumulation_steps
                train_loss_retain += accelerator.gather(loss_retain.repeat(args.train_batch_size)).mean().item() / args.gradient_accumulation_steps

                #########################################
                # Backpropagate
                #########################################
                
                # This is how it was before the munba trick:
                #accelerator.backward(loss)
                #if accelerator.sync_gradients:
                #    params_to_clip = lora_layers
                #    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                #optimizer.step()
                #lr_scheduler.step()
                #optimizer.zero_grad()
                
                # This is with the munba trick:
                
                # Compute gradients
                optimizer.zero_grad()
                accelerator.backward(loss_forget)
                grads_forget = [p.grad.clone() for p in unet.parameters() if p.requires_grad]  # This list has 256 elements; each element is a torch.Tensor of shapes like [4, 320], then [320, 4], then [4, 640], then [640, 4], etc
                
                optimizer.zero_grad()
                accelerator.backward(loss_retain)
                grads_retain = [p.grad.clone() for p in unet.parameters() if p.requires_grad]
                
                # TODO: append the cossine distance to the lists
                
                #for e in grads_forget:
                #    print(e.shape)


                # Stack gradients to form matrix G
                G = torch.stack([
                    torch.cat([g.view(-1) for g in grads_retain]),
                    torch.cat([g.view(-1) for g in grads_forget])
                ])
                K = G @ G.T  # Compute K = G^T G; It is a 2x2 tensor
                # K /= torch.norm(K)  # As recomended here: https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py#L231
                
                # Solve for α using narsh equation
                k11, k12, k22 = K[0, 0], K[0, 1], K[1, 1]
                alpha_retain = torch.sqrt((2 * k11 * k22 + k12 * torch.sqrt(k11 * k22)) / (k11**2 * k22 - k11 * k12**2))    # This is a Tensor of shape [], aka is a float
                alpha_forget = (1 - k11 * alpha_retain**2) / (k12 * alpha_retain)    
                alpha = torch.tensor([alpha_retain, alpha_forget]).reshape(2, 1)  # Typical values seem to be things like [0.0016, -0.0029]
                # print("Alpha in this iteration:", alpha)

                # Compute parameter update (but don't manually modify .data)
                G = G.to(accelerator.device)
                alpha = alpha.to(accelerator.device)
                
                scaled_grad = G.T @ alpha
                # scaled_grad = scaled_grad * 1/(2*alpha.min())
                scaled_grad /= torch.norm(alpha)

                # Assign updates as "fake gradients" for the optimizer
                for param, update in zip((p for p in unet.parameters() if p.requires_grad), 
                                         torch.split(scaled_grad, [p.numel() for p in unet.parameters() if p.requires_grad])):
                    param.grad = update.view(param.shape)  # Overwrite gradients

                # Gradient clipping
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    
                #########################################
                # End of Backpropagate
                #########################################

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss_forget": train_loss_forget}, step=global_step)
                accelerator.log({"train_loss_retain": train_loss_retain}, step=global_step)
                train_loss_forget = 0.0
                train_loss_retain = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss_forget.detach().item(), "step_loss_forget": loss_forget.detach().item(), "step_loss_retain": loss_retain.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the lora layers
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        t3 = time.time()

        # Final inference
        # Load previous pipeline
        if args.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline.load_lora_weights(args.output_dir)  # load attention processors
            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)  # run inference


        #################################
        pipeline_original, pipeline_learned, pipeline_unlearned = unlearn_lora(args.pretrained_model_name_or_path, args.output_dir, device=accelerator.device)
        eval_results, images2 = eval_text_to_image_unlearning(
            pipeline_original,
            pipeline_learned,
            pipeline_unlearned,
            eval_prompts_forget,
            eval_prompts_retain,
            judge_clip=ImageTextSimilarityJudge(metrics=['clip']),
        )
        images += [images2[path] for path in images2]

        t4 = time.time()


        metric_common_attributes = {
            "task_type": "text-to-image",
            "dataset_type": f"forget-and-retain-together",
            "dataset_name": f"{args.dataset_forget_name} (forget) and {args.dataset_retain_name} (retain) sets",
        }

        eval_results.append(EvalResult(
            metric_type = 'runtime',
            metric_name = f'Runtime init seconds (~↓)',
            metric_value = t1-t0,
            **metric_common_attributes,
        ))
        eval_results.append(EvalResult(
            metric_type = 'runtime',
            metric_name = f'Runtime data loading seconds (~↓)',
            metric_value = t2-t1,
            **metric_common_attributes,
        ))
        eval_results.append(EvalResult(
            metric_type = 'runtime',
            metric_name = f'Runtime training seconds (↓)',
            metric_value = t3-t2,
            **metric_common_attributes,
        ))
        eval_results.append(EvalResult(
            metric_type = 'runtime',
            metric_name = f'Runtime eval seconds (~↓)',
            metric_value = t4-t3,
            **metric_common_attributes,
        ))

        ################################
        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_forget_name=args.dataset_forget_name,
                dataset_retain_name=args.dataset_retain_name,
                repo_folder=args.output_dir,
                eval_results=eval_results,
                tags = [
                    "stable-diffusion",
                    "stable-diffusion-diffusers",
                    "text-to-image",
                    "diffusers",
                    "diffusers-training",
                    "lora",
                ],
                # TODO: pass the cossine distances
            )

            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
