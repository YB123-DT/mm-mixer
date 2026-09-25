from torch import nn


class SubspaceTokenEncoder(nn.Module):
    """Verified E14 K4/d128/L1 implementation with legacy-compatible keys."""
    def __init__(self, dropout: float):
        super().__init__()
        self.token_count = 4
        self.token_dim = 128
        self.split = nn.Linear(256, 4 * 128)
        layer = nn.TransformerEncoderLayer(
            128, 8, 512, dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.output_adapter = nn.Linear(128, 256)

    def forward(self, x):
        batch, modalities, _ = x.shape
        tokens = self.split(x).reshape(batch, modalities * 4, 128)
        encoded = self.transformer(tokens)
        grouped = encoded.reshape(batch, modalities, 4, 128).mean(dim=2)
        return self.output_adapter(grouped)


class ThreeTokenWideEncoder(nn.Module):
    """Experiment 45 parameter-matched three-token control (d256/FFN192)."""
    def __init__(self, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            256, 8, 192, dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 1)

    def forward(self, x):
        return self.transformer(x)
