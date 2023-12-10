import torch
from torch import nn

from model.block import DownsampleBlock


class EnhancedDiscriminator(nn.Module):
    def __init__(self, writer_count, character_count):
        super(EnhancedDiscriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            DownsampleBlock(1, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 512),
            DownsampleBlock(512, 512),
            nn.Flatten(),
        )

        self.reality_classifier = nn.Linear(512 * 4 * 4, 1)
        self.writer_classifier = nn.Linear(512 * 4 * 4, writer_count)
        self.character_classifier = nn.Linear(512 * 4 * 4, character_count)

    def forward(self, input):
        feature = self.feature_extractor(input)

        reality = torch.sigmoid(self.reality_classifier(feature))
        writer = self.writer_classifier(feature)
        character = self.character_classifier(feature)
        
        return reality, writer, character
