import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_utils import NUM_CLASSES


# ------------------------------------------------------------------
# 1.  Baseline GRU Classifier
# ------------------------------------------------------------------

class GRUClassifier(nn.Module):
    """
    Bidirectional multi-layer GRU for sequence classification.

    Architecture
    ------------
    Input  (B, T, F)
        → LayerNorm
        → BiGRU  ×  num_layers
        → Attention-pooling over T
        → Dropout  →  Linear  →  logits (B, C)

    Parameters
    ----------
    input_size   : number of input features F
    hidden_size  : GRU hidden units per direction
    num_layers   : stacked GRU layers
    num_classes  : output classes (default = 10)
    dropout      : dropout between GRU layers and before FC
    bidirectional: use bidirectional GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float   = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size   = hidden_size

        self.norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * self.num_directions

        # Learnable temporal attention
        self.attn = nn.Linear(gru_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, T, F)
        lengths : (B,) original sequence lengths (optional, for masking)

        Returns
        -------
        logits  : (B, num_classes)
        """
        x = self.norm(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)         # (B, T, D)

        # Temporal attention pooling
        attn_w = self.attn(out)          # (B, T, 1)
        if lengths is not None:
            # mask padding positions
            mask = _length_mask(lengths, out.size(1), x.device)
            attn_w = attn_w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * out).sum(dim=1)  # (B, D)

        return self.classifier(context)


# ------------------------------------------------------------------
# 2.  CNN-GRU Classifier  (Bonus)
# ------------------------------------------------------------------

class CNNGRUClassifier(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float   = 0.3,
        cnn_channels: tuple = (32, 64, 128),
    ):
        super().__init__()

        # --- CNN blocks ---
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),   # pool only in frequency
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute the feature dimension after CNN
        F_after = input_size
        for _ in cnn_channels:
            F_after = F_after // 2
        gru_in = cnn_channels[-1] * F_after

        self.norm = nn.LayerNorm(gru_in)

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        gru_out_dim = hidden_size * 2
        self.attn   = nn.Linear(gru_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        B, T, F = x.shape
        # (B, 1, T, F)
        x_cnn = x.unsqueeze(1)
        x_cnn = self.cnn(x_cnn)          # (B, C, T, F')
        C, T2, F2 = x_cnn.shape[1], x_cnn.shape[2], x_cnn.shape[3]
        # merge channel and frequency: (B, T2, C*F2)
        x_seq = x_cnn.permute(0, 2, 1, 3).reshape(B, T2, C * F2)
        x_seq = self.norm(x_seq)

        out, _ = self.gru(x_seq)         # (B, T2, D)

        attn_w = torch.softmax(self.attn(out), dim=1)
        context = (attn_w * out).sum(dim=1)

        return self.classifier(context)
# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _length_mask(lengths: torch.Tensor, max_len: int,
                 device: torch.device) -> torch.Tensor:
    """Return a boolean mask (B, T) — True for valid positions."""
    B = lengths.size(0)
    return torch.arange(max_len, device=device).unsqueeze(0) < \
           lengths.unsqueeze(1)


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

MODEL_REGISTRY = {
    "gru":         GRUClassifier,
    "cnn_gru":     CNNGRUClassifier,
}


def build_model(name: str, input_size: int, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name       : "gru" | "cnn_gru" | "attn_gru"
    input_size : number of input features F
    **kwargs   : forwarded to the model constructor

    Returns
    -------
    nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_size=input_size, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, T, F = 8, 173, 197   # typical for combined features, 4-s clip

    for name, cls in MODEL_REGISTRY.items():
        if name == "cnn_gru":
            m = cls(input_size=F)
        else:
            m = cls(input_size=F)
        x = torch.randn(B, T, F)
        y = m(x)
        print(f"{name:<12s}  output: {tuple(y.shape)}  "
              f"params: {count_parameters(m):,}")
