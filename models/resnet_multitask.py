from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_PHASE_CLASSES, NUM_TOOL_CLASSES


class TemporalAveragePool(nn.Module):
    """Temporal average pooling over a short frame window.

    Expects inputs of shape (B, T, C) and returns (B, C).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class PhaseToolNet(nn.Module):
    """ResNet-18 backbone with dual heads for phase and tool prediction.

    Phase logits are produced first and converted to probabilities; these
    probabilities are concatenated with the visual features and passed to the
    tool head so that tool predictions are explicitly conditioned on phase.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()

        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.temporal_pool = TemporalAveragePool()

        self.phase_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, NUM_PHASE_CLASSES),
        )

        self.tool_head = nn.Sequential(
            nn.Linear(feature_dim + NUM_PHASE_CLASSES, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, NUM_TOOL_CLASSES),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            frames: Tensor of shape (B, T, C, H, W).

        Returns:
            phase_logits: (B, num_phases)
            tool_logits: (B, num_tools)
        """

        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)

        feats = self.backbone(frames)  # (B*T, F)
        feats = feats.view(b, t, -1)   # (B, T, F)

        video_feat = self.temporal_pool(feats)  # (B, F)

        phase_logits = self.phase_head(video_feat)
        phase_probs = torch.softmax(phase_logits, dim=-1)

        tool_input = torch.cat([video_feat, phase_probs], dim=-1)
        tool_logits = self.tool_head(tool_input)

        return phase_logits, tool_logits
