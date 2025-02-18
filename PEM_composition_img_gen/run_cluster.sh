#!/bin/sh
#SBATCH --job-name=unlearn_pem_train                # Name of the job
#SBATCH --partition=gpu                             # Use the GPU partition
#SBATCH --gres=gpu:v100:1	                        # Select 1 V100 GPU
#SBATCH --time=3:00:00                              # Set maximum run time for the job (hh:mm:ss)
#SBATCH --output=assets/cluster_jobs/%j-train-out   # Redirect stdout
#SBATCH --error=assets/cluster_jobs/%j-train-err    # Redirect stderr

if [ "$(basename "$PWD")" != "PEM_composition_img_gen" ]; then
    echo "Error: This script must be run from the 'PEM_composition_img_gen' directory."
    exit 1
fi

srun apptainer exec --nv run_cluster.sif accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name="lambdalabs/naruto-blip-captions"\
    --dataloader_num_workers=8 \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --mixed_precision "fp16" \
    --train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir="./assets/lora/naruto" \
    --push_to_hub \
    --hub_model_id="LeonardoBenitez/sd-lora-naruto" \
    --num_train_epochs=1 \
    --validation_epochs=3 \
    --checkpointing_steps=500 \
    --validation_prompt="A naruto with blue eyes." \
    --seed=42