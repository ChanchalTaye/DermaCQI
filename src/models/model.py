# ====================== model.py ======================

import os
import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


class MobileNetV3Classifier(nn.Module):

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        weight_path=None
    ):
        super().__init__()

        # ======================================================
        # Local pretrained weight path
        # ======================================================

        if weight_path is None:
            weight_path = r"D:\Projects\DERMACQI\MobileNetV3_HAM10000\mobiletv3_large.safetensors"

        print("=" * 70)
        print("Initializing MobileNetV3 Large")
        print(f"Classes      : {num_classes}")
        print(f"Pretrained   : {pretrained}")
        print("=" * 70)

        # ======================================================
        # Backbone
        # ======================================================

        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        # ======================================================
        # Load pretrained weights
        # ======================================================

        if pretrained:

            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"\nPretrained weight not found:\n{weight_path}"
                )

            print("\nLoading pretrained weights...")

            state_dict = load_file(weight_path)

            state_dict = {
                k.replace("model.", "").replace("module.", ""): v
                for k, v in state_dict.items()
            }

            msg = self.backbone.load_state_dict(
                state_dict,
                strict=False
            )

            print(f"Missing Keys    : {len(msg.missing_keys)}")
            print(f"Unexpected Keys : {len(msg.unexpected_keys)}")

        else:
            print("\nTraining from scratch.")

        # ======================================================
        # Classification Head
        # ======================================================

        self.head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(1280, num_classes)
        )

        print("\nModel Ready!")

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x):

        features = self.backbone(x)

        logits = self.head(features)

        return logits


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    model = MobileNetV3Classifier(
        num_classes=7,
        pretrained=True
    )

    x = torch.randn(2, 3, 384, 384)

    with torch.no_grad():
        y = model(x)

    print("\nInput :", x.shape)
    print("Output:", y.shape)