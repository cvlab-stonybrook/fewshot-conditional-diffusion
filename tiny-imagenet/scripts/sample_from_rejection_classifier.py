import numpy as np
import torch
import torch.nn.functional as F
import argparse

import tqdm

from guided_diffusion.script_util import create_model_and_diffusion,

from pathlib import Path
import gc
from guided_diffusion import dist_util
import torch as th
import torch.nn as nn
import time
import uuid

device = torch.device(f"cuda:0")
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output  # .detach()

    return hook


class CNN_simple(nn.Module):
    def __init__(self, in_ch=512, hidden=128, num_classes=200):
        super(CNN_simple, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden, hidden, 3, padding=1)

        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm1 = nn.GroupNorm(32, hidden)
        self.norm2 = nn.GroupNorm(32, hidden)
        self.norm3 = nn.GroupNorm(32, hidden)
        self.norm4 = nn.GroupNorm(32, hidden)

        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        x = self.mp1(x)
        x = self.norm3(F.relu(self.conv3(x)))
        x = self.norm4(F.relu(self.conv4(x)))
        x = self.mp2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_diff_features(diff_net, img, t1, t2, y=None):
    img = img.float().to(device)
    t1_tensor = torch.tensor([t1]).float().to(device)
    t2_tensor = torch.tensor([t2]).float().to(device)
    x = diff_net(img, t1_tensor, y)

    input_features1 = features["middle"]

    x = diff_net(img, t2_tensor, y)

    input_features2 = features["middle"]

    input_features = torch.cat((input_features1, input_features2), dim=1)
    return input_features


uncond_path = Path("checkpoints_openai/imagenet64_uncond_100M_1500K.pt")
cond_path = Path(f"logs/tiny_imagenet_class_cond/ema_0.9999_120_000.pt")

args_diff = {
    "image_size": 64,
    "num_channels": 128,
    "num_res_blocks": 3,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_head_channels": -1,
    "attention_resolutions": "16,8",
    "channel_mult": "",
    "dropout": 0,
    "class_cond": True,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
    "resblock_updown": False,
    "use_fp16": False,
    "use_new_attention_order": False,
    "learn_sigma": True,
    "diffusion_steps": 1000,
    "noise_schedule": "cosine",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
    "num_classes": 200,
    "timestep_respacing": "ddim100",
}


uncond_model, diffusion = create_model_and_diffusion(**{**args_diff, **{"class_cond": False}})
uncond_model.load_state_dict(dist_util.load_state_dict(str(uncond_path), map_location="cpu"))

cond_model, diffusion = create_model_and_diffusion(**args_diff)
cond_model.load_state_dict(dist_util.load_state_dict(str(cond_path), map_location="cpu"))
middle_hook = uncond_model.middle_block.register_forward_hook(get_features("middle"))

list(map(lambda x: x.to(device), [uncond_model, cond_model]))
list(map(lambda x: x.eval(), [uncond_model, cond_model]))
cnn = CNN_simple(1024, hidden=256).to(device)
cnn.load_state_dict(torch.load("<path_to_rejection_classifier>.pth", map_location=device))
cnn.eval()


def get_samples(w, classes, diffusion, use_ddim=False):
    batch_size = len(classes)
    classes = th.tensor(classes, device=device)
    model_kwargs = {"y": classes}

    sample_fn = diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop

    sample = sample_fn(
        cond_model,
        (batch_size, 3, 64, 64),
        clip_denoised=True,
        model_kwargs=model_kwargs,
        classifier_free_guidance=True,
        guidance_strength=w,
    )
    t1, t2 = 300, 800

    with torch.no_grad():
        input_features = get_diff_features(uncond_model, sample, t1, t2)
        # Predict logits
        pred = cnn(input_features)
    probs = F.softmax(pred, dim=-1)[range(len(pred)), classes.long()]
    probs = probs.contiguous().cpu().numpy()

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous().cpu().numpy()

    return sample, probs


def main(args):
    thresh = args.threshold
    n_imgs = args.n_images

    bs = args.batch_size
    target_class = args.target_class

    w = args.guidance_str

    print(f"Sampling images of class {target_class}")

    t = time.perf_counter()

    res_folder = Path(args.target_dir) / str(args.target_class)
    res_folder.mkdir(exist_ok=True, parents=True)

    x_lis = []
    prob_lis = []

    pbar = tqdm.tqdm(total=n_imgs)
    it = 0
    classes = [target_class] * bs
    while len(prob_lis) < n_imgs:
        it += 1
        x, prob = get_samples(w, classes, diffusion, use_ddim=True)

        prob_good = prob[prob > thresh]
        if len(prob_good):
            # samples are accepted by the rejection classifier 

            n = len(prob_good)
            x_good = x[prob > thresh]

            x_lis.append(x_good)
            prob_lis.extend(prob_good.tolist())
            pbar.update(n)


    pbar.close()

    x_arr = np.concatenate(x_lis, axis=0)
    uniq = uuid.uuid4().hex[:8]

    n_imgs_out = len(x_arr)

    if n_imgs_out == 0:
        return

    np.save(str(res_folder / f"{n_imgs_out}_{uniq}.npy"), x_arr)
    np.save(str(res_folder / f"{n_imgs_out}_{uniq}_prob.npy"), prob_lis)

    print(f"Sampled {n_imgs_out} for class {target_class}")
    print(f"Used {it} batches in total")
    print(f"Took {(time.perf_counter() - t)//60} min \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--guidance_str", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
