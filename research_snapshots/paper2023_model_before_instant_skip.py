"""Borghesi et al. 2023 two-branch trajectory TCN + ball-query attention.

Independent PyTorch implementation of the published architecture. The choices
not fixed in the paper and external-data adaptations are in PAPER_2023_PROTOCOL.md.
"""
import math
import numpy as np
from torch_runtime import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, 2, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        x = F.relu(self.conv(F.pad(x, (self.dilation, 0))))
        return F.relu(self.mix(self.drop(x)))


class TrajectoryEncoder(nn.Module):
    def __init__(self, width=64, dropout=.1, channels=11):
        super().__init__()
        self.stem = nn.Conv1d(channels, width, 1)
        self.first_norm = nn.BatchNorm1d(width, eps=.001, momentum=.01)
        self.blocks = nn.Sequential(*[CausalBlock(width, 2**i, dropout) for i in range(6)])
        self.drop = nn.Dropout(dropout)
        self.last_norm = nn.BatchNorm1d(width, eps=.001, momentum=.01)

    def forward(self, x):
        # Flattening batch and object dimensions is equivalent to a shared kx1
        # convolution, with normalization across batch, object and time axes.
        batch, frames, objects, channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch*objects, channels, frames)
        z = self.first_norm(F.relu(self.stem(x)))
        z = self.last_norm(self.drop(self.blocks(z)))
        return z.reshape(batch, objects, -1, frames).permute(0, 3, 1, 2)


class BallAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)

    def forward(self, objects):
        q = self.query(objects[..., 0:1, :])
        k, v = self.key(objects), self.value(objects)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        return torch.matmul(score.softmax(-1), v).squeeze(-2)


class Branch(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = BallAttention(width)
        self.classifier = nn.Sequential(nn.Linear(width, 64), nn.ELU(),
                                        nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.classifier(self.attention(z)).squeeze(-1)


class PaperPossession(nn.Module):
    def __init__(self, width=64, dropout=.1, channels=11):
        super().__init__()
        self.encoder = TrajectoryEncoder(width, dropout, channels)
        self.dead = Branch(width)
        self.home = Branch(width)
        self.apply(self._initialize)
        self.reset_attention_initialization()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_attention_initialization(self):
        # Attention projections use Xavier normal; FFN projections use uniform.
        for branch in [self.dead, self.home]:
            for m in [branch.attention.query, branch.attention.key, branch.attention.value]:
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        z = self.encoder(x)[:, -1]
        return self.dead(z), self.home(z)

    def dense(self, x):
        z = self.encoder(x)
        return self.dead(z), self.home(z)


def joint_loss(dead, home, labels, weights=None, alpha=.5):
    valid = labels < 3
    alive = labels < 2
    dead_loss = F.binary_cross_entropy_with_logits(dead, (labels == 2).float(), reduction='none')
    home_loss = F.binary_cross_entropy_with_logits(home, (labels == 0).float(), reduction='none')
    loss = alpha*dead_loss + (1-alpha)*home_loss*alive
    if weights is not None:
        loss = loss*weights
    return loss[valid].mean()


def outputs(model, x, device, chunk=1024):
    """Return dead/home marginals at 16 Hz with exact sliding-window alignment."""
    model.eval()
    result = np.empty((len(x), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            end = min(len(x), start+chunk)
            # Label t consumes [t-47, ..., t+16]. Network output at its final
            # input index uses 64 preceding samples (receptive field 64).
            ix = np.clip(np.arange(start-47, end+16), 0, len(x)-1)
            a, b = model.dense(torch.from_numpy(x[ix]).to(device)[None])
            result[start:end, 0] = a[0, 63:63+end-start].sigmoid().cpu().numpy()
            result[start:end, 1] = b[0, 63:63+end-start].sigmoid().cpu().numpy()
    return result


def upsample(marginals, samples, n):
    return np.column_stack([np.interp(np.arange(n), samples, marginals[:, k])
                            for k in [0, 1]]).astype(np.float32)


def joint_probabilities(marginals):
    dead, home = marginals.T
    return np.column_stack([(1-dead)*home, (1-dead)*(1-home), dead])


def hierarchical_labels(marginals):
    return np.where(marginals[:, 0] >= .5, 2,
                    np.where(marginals[:, 1] >= .5, 0, 1)).astype(np.int8)
