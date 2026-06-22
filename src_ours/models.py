import torch
import torch.nn as nn

from src_ours.constants import TEST_DATASET_FOLDERS, TRAIN_DATASET_FOLDERS


class DummyModelDictInput(nn.Module):
    """
    Always predict 0.
    """
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        x = next(iter(batch.values()))
        batch_size = x.shape[0]

        return (
            torch.zeros(
                (batch_size, 4, 256, 256),
                device=x.device,
                dtype=x.dtype,
            )
            + self.dummy
        )

# ==================

class PixelWiseBaseline(nn.Module):
    """
    Pixel-wise baseline model (1x1 Conv MLP).

    Works with MultiFolderNpyDataset output:
        sample = (
            {
                "alphaearth_emb": Tensor(C,H,W),
                "s2": Tensor(C,H,W),
                ...
            }, 
            "label": Tensor(C,H,W)
        )

    Only one input source is used (specified by input_key).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        input_key,
        hidden_dim=16,
    ):
        super().__init__()

        self.input_key = input_key

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, batch):
        """
        batch: dict from DataLoader
            {
                input_key: (B, C, H, W),
                other_keys: ignored,
                label: optional
            }
        """
        keys = batch.keys()

        if TEST_DATASET_FOLDERS[self.input_key] in keys:
            current_input_key = TEST_DATASET_FOLDERS[self.input_key]
        elif TRAIN_DATASET_FOLDERS[self.input_key] in keys:
            current_input_key = TRAIN_DATASET_FOLDERS[self.input_key]

        x = batch[current_input_key]  # select ONLY one modality
        return self.net(x)
    

# ========== LIGHT UNET with second input in the middle

import torch
import torch.nn as nn

from src_ours.constants import TEST_DATASET_FOLDERS, TRAIN_DATASET_FOLDERS


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DualInputLightUNet(nn.Module):

    VALID_MAIN_INPUTS = {"alphaearth", "tessera"}

    VALID_AUX_INPUTS = {
        "terraminds1",
        "terraminds2",
        "thors1",
        "thors2",
    }

    def __init__(
        self,
        main_input_name,
        aux_input_name,
        main_in_channels=None, # alphaearth 64 or tessera 128
        aux_in_channels=768,
        n_classes=4,
    ):
        super().__init__()

        if main_in_channels is None:
            if main_input_name == "alphaearth":
                main_in_channels = 64
            elif main_input_name == "tessera":
                main_in_channels = 128
            else:
                raise ValueError("Model input channels of main image not specified")

        if main_input_name not in self.VALID_MAIN_INPUTS:
            raise ValueError(
                f"main_input_name must be one of "
                f"{sorted(self.VALID_MAIN_INPUTS)}"
            )

        if aux_input_name not in self.VALID_AUX_INPUTS:
            raise ValueError(
                f"aux_input_name must be one of "
                f"{sorted(self.VALID_AUX_INPUTS)}"
            )

        self.main_train_key = TRAIN_DATASET_FOLDERS[main_input_name]
        self.main_test_key = TEST_DATASET_FOLDERS[main_input_name]

        self.aux_train_key = TRAIN_DATASET_FOLDERS[aux_input_name]
        self.aux_test_key = TEST_DATASET_FOLDERS[aux_input_name]

        # Main encoder
        # 256x256x(64/128) -> 32x32x256
        self.inc = DoubleConv(main_in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Auxiliary branch
        # 16x16x768 -> 32x32x128
        self.aux_encoder = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.Conv2d(aux_in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # 32x32x256 -> 256x256x4
        
        # Bottleneck:
        # 256 + 128 = 384 channels
        self.up1 = UpsampleBlock(384, 128)

        self.conv1 = DoubleConv(
            128 + 128,  # upsample + skip
            128,
        )
        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(64 + 64, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(32 + 32, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def _get_input(self, batch, train_key, test_key):
        if train_key in batch:
            return batch[train_key]

        if test_key in batch:
            return batch[test_key]

        raise KeyError(
            f"Neither '{train_key}' nor '{test_key}' found. "
            f"Available keys: {list(batch.keys())}"
        )
    
    def forward(self, batch):

        x_main = self._get_input( batch, self.main_train_key, self.main_test_key)

        x_aux = self._get_input( batch, self.aux_train_key, self.aux_test_key)
        # Main encoder
        x1 = self.inc(x_main)       # 256x256x32
        x2 = self.down1(x1)         # 128x128x64
        x3 = self.down2(x2)         # 64x64x128
        x4 = self.down3(x3)         # 32x32x256

        # Auxiliary branch
        aux = self.aux_encoder(x_aux)   # 32x32x128

        # Fuse bottleneck
        x = torch.cat([x4, aux], dim=1)  # 32x32x384

        # Decoder
        x = self.up1(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)

        return self.outc(x)