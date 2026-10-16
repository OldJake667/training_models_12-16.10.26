"""Original CNN and grayscale ViT-Tiny architectures, formatted for inspection."""
import torch
from torch import nn


class ScratchCNN(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 30 * 30, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=1, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size} x {self.image_size} images")
        return self.projection(images).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dimension, hidden_dimension, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden_dimension), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension), nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return self.layers(tokens)


class TransformerBlock(nn.Module):
    def __init__(self, dimension=192, heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, dimension * mlp_ratio, dropout)

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class VisionTransformerTiny(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        dimension = 192
        self.patch_embedding = PatchEmbedding(224, 16, 1, dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position_embedding = nn.Parameter(torch.zeros(1, 197, dimension))
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(dimension, heads=3, mlp_ratio=4, dropout=0.1) for _ in range(12)
        ])
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        tokens = self.patch_embedding(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = self.dropout(torch.cat((class_token, tokens), dim=1) + self.position_embedding)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])
