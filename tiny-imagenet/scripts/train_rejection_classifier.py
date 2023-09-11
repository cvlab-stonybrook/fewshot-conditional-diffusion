import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import tqdm

from guided_diffusion.unet import UNetModel
from torch.utils.data import DataLoader


from guided_diffusion.image_datasets import _list_image_files_recursively, ImageDataset
import blobfile as bf
import os

device = torch.device("cuda:0")


def get_dataset(folder, random_flip=True):
    """Get image dataset from folder"""

    all_files = _list_image_files_recursively(folder)

    class_names = [bf.basename(path).split("_")[0] for path in all_files]

    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    img_paths = np.array(all_files, dtype=np.string_)

    dataset = ImageDataset(64, img_paths, classes=classes, random_flip=random_flip)
    return dataset


# Load unconditional diffusion model
diff_net = UNetModel(
    image_size=64,
    in_channels=3,
    out_channels=6,
    model_channels=128,
    num_res_blocks=3,
    channel_mult=(1, 2, 3, 4),
    attention_resolutions=[4, 8],
    num_head_channels=-1,
    dropout=0.0,
    resblock_updown=False,
    use_scale_shift_norm=True,
    num_heads=4,
    use_new_attention_order=False,
).to(device)

diff_net.load_state_dict(torch.load("checkpoints_openai/imagenet64_uncond_100M_1500K.pt", map_location=device))
print("Loaded unconditional Diffusion Model")
print(f"# parameters: {sum([p.numel() for p in diff_net.parameters()]):,}")

# Add hooks
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output  # .detach()

    return hook


middle_hook = diff_net.middle_block.register_forward_hook(get_features("middle"))


# Construct classifier
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

        self.dropout = nn.Dropout(0.2)

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
        x = self.dropout(x)
        x = self.classifier(x)

        return x


cnn = CNN_simple(1024, hidden=256).to(device)
total_params = sum(p.numel() for p in cnn.state_dict().values())
print(f"Total number of parameters: {total_params:,}")


# Get features from U-Net
def get_diff_features(diff_net, img, t1, t2, y=None):
    img = img.float().to(device)
    t1_tensor = torch.tensor([t1]).float().to(device)
    t2_tensor = torch.tensor([t2]).float().to(device)
    _ = diff_net(img, t1_tensor, y)

    input_features1 = features["middle"]

    _ = diff_net(img, t2_tensor, y)

    input_features2 = features["middle"]
    input_features = torch.cat((input_features1, input_features2), dim=1)

    input_features_norm = F.normalize(input_features, dim=1)
    return input_features_norm


opt = torch.optim.Adam(cnn.parameters(), lr=3e-4, weight_decay=0.0001)

diff_net.eval()
criterion = nn.CrossEntropyLoss()
epochs = 30
update_every = 20


train_dataset = get_dataset("datasets/tiny-imagenet-200/train/")
batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
val_dataset = get_dataset("datasets/tiny-imagenet-200/val/images_folders", random_flip=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

diffusion_steps = 1000
t1, t2 = 0.3 * diffusion_steps, 0.8 * diffusion_steps


def get_val_loss(cnn):
    cnn.eval()
    val_loss = []
    n_correct, N = 0, 0
    for batch in val_dataloader:
        img, labels = batch

        with torch.no_grad():
            y = labels["y"].long().to(device)
            input_features = get_diff_features(diff_net, img, t1, t2)

            # Predict logits
            pred = cnn(input_features)

            # Compute loss
            loss = criterion(pred, y)
            val_loss.append(loss.item())
            pred_lab = torch.argmax(pred, axis=1)
            n_correct += int((pred_lab == y).sum())
            N += len(y)
    print(f"Val accuracy: {(n_correct / N):.3f}")

    return val_loss


min_val_loss = 1000
model_path = f"rejection/{diffusion_steps}_decay_1e-4/"
os.makedirs(model_path, exist_ok=True)

model_scripted = torch.jit.script(cnn)  # Export to TorchScript
model_scripted.save(f"{model_path}/net.pth")  # Save


for e in range(epochs):
    cnn.train()
    print(f"Epoch [{e+1:02d}/{epochs:02d}]")
    train_loss = []

    batch_bar = tqdm.tqdm(train_dataloader)
    for i, batch in enumerate(batch_bar):
        img, labels = batch

        # Get features from U-Net
        with torch.no_grad():
            y = labels["y"].long().to(device)
            input_features = get_diff_features(diff_net, img, t1, t2)

        # Predict logits
        pred = cnn(input_features)

        # Compute loss
        loss = criterion(pred, y)

        # Update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())
        if i % update_every == 0:
            batch_bar.set_postfix({"Loss": np.mean(train_loss[-update_every:])})

    val_loss = np.mean(get_val_loss(cnn))

    print(
        f"Valid loss: {val_loss:.3f}",
    )
    print(f"Train loss: {np.mean(train_loss):.3f}")

    model_name = f"{model_path}/epoch_{e}_param_{total_params//10**3}K.pth"

    if val_loss < min_val_loss or e % 5 == 0:
        min_val_loss = val_loss
        torch.save(cnn.state_dict(), model_name)
