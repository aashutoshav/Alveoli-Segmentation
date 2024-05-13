import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.img_filenames = os.listdir(self.img_dir)
        self.msk_filenames = os.listdir(self.msk_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_filenames[index])
        msk_name = os.path.join(self.msk_dir, self.msk_filenames[index])

        image = self.transform(Image.open(img_name))
        mask = self.transform(Image.open(msk_name))

        return image, mask

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()

        self.layers = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        return self.layers(x)

class UNetR(nn.Module):
    def __init__(self, config):
        super(UNetR, self).__init__()
        self.config = config

        # Positional Embedding
        self.patch_embed = nn.Linear(
            config["n_channels"] * config["patch_size"] ** 2, config["embed_dim"]
        )

        self.positions = torch.arange(
            start=0, end=config["num_patches"], step=1, dtype=torch.int32
        )
        self.pos_embed = nn.Embedding(config["num_patches"], config["embed_dim"])

        self.transformer_encoder_layers = []

        for i in range(config["num_layers"]):
            self.transformer_encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    dim_feedforward=config["mlp_dim"],
                    dropout=config["dropout_rate"],
                    activation=nn.GELU(),
                    batch_first=True,
                )
            )

        """Decoder 1"""

        self.d1 = DeConvBlock(config["embed_dim"], 512)
        self.s1 = nn.Sequential(
            DeConvBlock(config["embed_dim"], 512), ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(ConvBlock(512 * 2, 512), ConvBlock(512, 512))

        """Decoder 2"""
        self.d2 = DeConvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeConvBlock(config["embed_dim"], 256),
            ConvBlock(256, 256),
            DeConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        self.c2 = nn.Sequential(ConvBlock(256 * 2, 256), ConvBlock(256, 256))

        """Decoder 3"""
        self.d3 = DeConvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeConvBlock(config["embed_dim"], 128),
            ConvBlock(128, 128),
            DeConvBlock(128, 128),
            ConvBlock(128, 128),
            DeConvBlock(128, 128),
            ConvBlock(128, 128),
        )
        self.c3 = nn.Sequential(ConvBlock(128 * 2, 128), ConvBlock(128, 128))

        """Decoder 4"""
        self.d4 = DeConvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        self.c4 = nn.Sequential(ConvBlock(64 * 2, 64), ConvBlock(64, 64))

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        batch, channels, width, height = inputs.shape
        inputs = inputs.view(batch, -1, self.config['n_channels'] * self.config['patch_size']**2)
        x = self.patch_embed(inputs)
        x = x + self.pos_embed(self.positions)

        skip_idx = [3, 6, 9, 12]
        skips = []

        for i in range(self.config["num_layers"]):
            layer = self.transformer_encoder_layers[i]
            x = layer(x)

            if i + 1 in skip_idx:
                skips.append(x)

        z3, z6, z9, z12 = skips
        z0 = inputs.view(
            batch,
            self.config["n_channels"],
            self.config["img_size"],
            self.config["img_size"],
        )
        shape = (
            batch,
            self.config["embed_dim"],
            self.config["patch_size"],
            self.config["patch_size"],
        )

        ## Decoder 1
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)

        ## Decoder 2
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        return self.output(x)

model = UNetR(
    {
        "n_channels": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "num_patches": 256,
        "num_layers": 12,
        "num_heads": 8,
        "mlp_dim": 2048,
        "dropout_rate": 0.1,
        "img_size": 256,
    }
)

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        torch.sum(y_true) + torch.sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


dataset = CustomImageDataset('images', 'masks')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = dice_loss(masks, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch: {epoch}, Training Dice Loss: {avg_loss}")
    
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for imgs, masks in val_loader:
            outputs = model(imgs)
            loss = dice_loss(masks, outputs)
            val_loss += loss.item() * imgs.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch: {epoch}, Validation Dice Loss: {avg_val_loss}")

torch.save(model.state_dict(), "model.pt")