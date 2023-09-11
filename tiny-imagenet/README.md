# Tiny Imagenet Experiments


This code is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

# Download pre-trained model

We have released checkpoint for the class conditional  Tiny imagenet model. 


 * 64x64 class conditional DM: [ema_0.9999_120000.pt](https://drive.google.com/file/d/1WWxoAS1rG1KinLiitEMHD_mRvtszzO9o/view?usp=sharing)

# Training

To train a conditional diffusion model, first download the Tiny Imagenet dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unconditional Imagenet checkpoint from the `openai/improved-diffusion` repo [here](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt) . Then store the dataset in a folder called `tiny-imagenet-200` in the `datasets` folder. Then run the following command:

```bash
python scripts/image_train.py --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine --lr 1e-4 --batch_size 128 --class_cond True --resume_checkpoint checkpoints_openai/imagenet64_uncond_100M_1500K.pt --data_dir datasets/tiny-imagenet-200/train --save_interval 20_000 --num_samples 32
```

To train the rejection classifier used in our experiments, run the following command after updating paths in the script:


```bash
python scripts/train_rejection_classifier.py
```

# Sampling

For vanilla conditional sampling, run the following command:

```bash
python scripts/image_sample.py <training_args> --use_ddim True --timestep_respacing ddim100
```

For rejection sampling with classifier-free guidance, run:
```bash
python scripts/sample_from_rejection_classifier.py --target_dir <dir_to_save_images> --threshold <rejection_threshold>  --target_class <target_class> --guidance_str <cf_guidance_str>
```
The above script uses ddim sampling with 100 steps.

For vanilla classifier-free guidance without rejection sampling, run the same command with `--threshold 0.0`.