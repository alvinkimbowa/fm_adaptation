"""The prompt that names the target, and the gates that carry it into the network.

The encoder is the swappable part. It turns one prompt per case into one vector per case, and that is
the whole contract: a text tower replacing the lookup table below is a change of class here and
nowhere else, because the index and the vocabulary are all either kind needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Index 0 means no prompt was given. The vocabulary occupies 1..len(vocabulary).
NULL_PROMPT = 0


class CategoricalPromptEncoder(nn.Module):
    """One trainable embedding per entry of the vocabulary, behind a null embedding at index 0."""

    def __init__(self, vocabulary, width: int):
        super().__init__()
        self.vocabulary = tuple(vocabulary)
        self.width = width
        self.embedding = nn.Embedding(len(self.vocabulary) + 1, width)

    def forward(self, prompt):
        return self.embedding(prompt)


class PromptGate(nn.Module):
    """Gates a feature map by the prompt: SwiGLU, with the gate drawn from the embedding.

    `out` is zero-initialised, so a freshly built gate is the identity and a prompted run starts from
    exactly the network it would have been without one.
    """

    def __init__(self, channels: int, width: int):
        super().__init__()
        self.gate = nn.Linear(width, channels)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features, embedding):
        gate = F.silu(self.gate(embedding))[:, :, None, None]
        return features + self.out(self.value(features) * gate)


class PromptConditioner(nn.Module):
    """The prompt encoder, and the gates sitting on the encoder's own feature maps.

    The decoder's gate is built by the decoder instead, so that it is saved and loaded with the rest
    of the head rather than needing a checkpoint key of its own.
    """

    def __init__(self, vocabulary, width: int, feature_channels, gate_features: bool):
        super().__init__()
        self.encoder = CategoricalPromptEncoder(vocabulary, width)
        self.gates = (
            nn.ModuleList(PromptGate(channels, width) for channels in feature_channels)
            if gate_features
            else None
        )

    def encode(self, prompt):
        return self.encoder(prompt)

    def condition(self, features, embedding):
        if self.gates is None:
            return features
        # The probes take a single map where the pyramid decoders take four.
        if isinstance(features, torch.Tensor):
            return self.gates[0](features, embedding)
        return [gate(feature, embedding) for gate, feature in zip(self.gates, features)]
