#!/bin/sh
#SBATCH --job-name=unlearn_pem_train                # Name of the job
#SBATCH --partition=gpu                             # Use the GPU partition
#SBATCH --gres=gpu:a100:1	                        # Select machine config (gpu:v100:1 = 1 V100 16GB GPU, gpu:a100:1 = 1 A100 40GB GPU)
#SBATCH --time=6:00:00                              # Set maximum run time for the job (hh:mm:ss)
#SBATCH --output=assets/cluster_jobs/%j-train-out   # Redirect stdout
#SBATCH --error=assets/cluster_jobs/%j-train-err    # Redirect stderr

if [ "$(basename "$PWD")" != "PEM_composition_img_gen" ]; then
    echo "Error: This script must be run from the 'PEM_composition_img_gen' directory."
    exit 1
fi

# srun apptainer exec --nv run_cluster.sif accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
#     --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
#     --dataset_name="lambdalabs/naruto-blip-captions"\
#     --dataloader_num_workers=2 \
#     --resolution=512 \
#     --center_crop \
#     --random_flip \
#     --mixed_precision "fp16" \
#     --train_batch_size=8 \
#     --gradient_accumulation_steps=4 \
#     --learning_rate=2e-04 \
#     --max_grad_norm=1 \
#     --lr_scheduler="cosine" \
#     --lr_warmup_steps=0 \
#     --output_dir="./assets/lora/naruto" \
#     --push_to_hub \
#     --hub_model_id="LeonardoBenitez/sd-lora-naruto" \
#     --num_train_epochs=30 \
#     --validation_epochs=3 \
#     --checkpointing_steps=500 \
#     --validation_prompt="A naruto with blue eyes." \
#     --seed=42


srun apptainer exec --nv run_cluster.sif accelerate launch --mixed_precision="fp16" --num_processes 2 --num_machines 1 --dynamo_backend no train_text_to_image_lora_munba.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_forget_name="lambdalabs/naruto-blip-captions"\
    --dataset_retain_name="Hamdy20002/COCO_Person"\
    --dataloader_num_workers=2 \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --mixed_precision "fp16" \
    --train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir="./assets/lora/naruto" \
    --push_to_hub \
    --hub_model_id="LeonardoBenitez/sd-lora-naruto" \
    --num_train_epochs=60 \
    --validation_epochs=10 \
    --checkpointing_steps=500 \
    --validation_prompt="A man with blue eyes." \
    --seed=42



