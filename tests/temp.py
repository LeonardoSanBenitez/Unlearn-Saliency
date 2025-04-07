from libs.metrics import FrechetInceptionDistance


for class_model in UnlearnDatasetImagenette.class_mapping:
    print('-'*80)
    print('Evaluating', UnlearnDatasetImagenette.class_mapping[class_model])
    pipeline_original, pipeline_learned, pipeline_unlearned = unlearn_lora(
        model_original_id = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        model_lora_id = f"./assets/lora/imagenette_splits/{class_model}",
        device = 'cuda',
    )

    # retain data
    ret_imgs: List[torch.Tensor]
    # forget data
    forget_imgs: List[torch.Tensor]

    for col, class_eval in enumerate(UnlearnDatasetImagenette.class_mapping):
        prompt = f"An image of {UnlearnDatasetImagenette.class_mapping[class_eval]}"

        if class_eval != class_model:
            ret_imgs.append(pipeline_unlearned(prompt, num_images_per_prompt=10).images)
        else:
            forget_imgs.append(pipeline_unlearned(prompt, num_images_per_prompt=10).images)
    
    metric_ret = FrechetInceptionDistance(metrics = ['FID'],
        real_imgs_path = f"./assets/lora/imagenette_splits/{class_model}/test_retain/",
        gen_imgs = ret_imgs
    ).score()
    print(f"FID for retain dataset: {metric_ret['FID']:.5f}")

    metric_forget = FrechetInceptionDistance(metrics = ['FID'],
        real_imgs_path = f"./assets/lora/imagenette_splits/{class_model}/test_forget/",
        gen_imgs = forget_imgs
    ).score()
    print(f"FID for forget dataset: {metric_forget['FID']:.5f}")
    #break
