"""
Train a diffusion model on images.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

import wandb
import os
from matplotlib import pyplot as plt


def generate_samples(args, data, diffusion, model, save_path):
    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
        sample_fn = diffusion.p_sample_loop
        with th.no_grad():
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        logger.log(f"saving to {save_path}")
        for i in range(len(arr)):
            im = Image.fromarray(arr[i])

            if args.class_cond:
                name = save_path / f"{i}_{label_arr[i]}.png"
            else:
                name = save_path / f"{i}.png"
            im.save(name)

    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(4, 4),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for ax, im, lab in zip(grid, arr[:16], label_arr[:16]):
        # Iterating over the grid returns the Axes.
        cls_text = str(lab)
        cls_text = " ".join(cls_text.split(" ")[:2])
        ax.imshow(im)
        ax.yaxis.set_visible(False)
        ax.set_title(f"{cls_text}")

    wandb.log({str(save_path.name): fig})
    dist.barrier()
    logger.log("sampling complete")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.use_mpi)
    logger.configure()
    logger.log("creating data loader...")
    num_classes, data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        labeled_idx_path=args.labeled_idx_path,
        data_type=args.data_type,
    )
    if args.classifier_free_guidance:
        # Last index is used for null token (no conditioning)
        num_classes += 1

    args.num_classes = num_classes

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **{**args_to_dict(args, model_and_diffusion_defaults().keys()), "num_classes": num_classes}
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    total_params = sum(p.numel() for p in model.state_dict().values())
    logger.log(f"Number of parameters in the model: {total_params:,}")

    if dist.get_rank() == 0:
        if args.wandb_name:
            wandb_args = {
                "entity": "<wandb-username>",
                "project": "<wandb-project>",
            }
            if os.environ.get("WANDB_RESUME", None):
                wandb_args["resume"] = "must"
                wandb_args["id"] = args.wandb_name
            else:
                wandb_args["name"] = args.wandb_name
            logger.log("wandb args: ", wandb_args)
            wandb.init(**wandb_args)

    # Generate samples every 10k steps
    sampling_function = lambda x, y, z, w: generate_samples(args, x, y, z, w) if args.num_samples else None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sampling_function=sampling_function,
        num_steps=args.num_steps,
        classifier_free_guidance=args.classifier_free_guidance,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_mpi=False,
        wandb_name="",
        num_samples=16,
        num_steps=300_000,
        labeled_idx_path="",
        classifier_free_guidance=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
