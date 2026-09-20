"""The prompt that names the target, and the gates that carry it into the network.

The encoder is the swappable part. It turns one prompt per case into one vector per case, and that is
the whole contract: a text tower replacing the lookup table below is a change of class here and
nowhere else, because the index and the vocabulary are all either kind needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalPromptEncoder(nn.Module):
    """One trainable embedding per vocabulary entry, behind a null embedding per field.

    The fields share one table, each holding a contiguous block of rows that opens with its own null.
    A prompt is a row of indices into that table: one column for a field answered with a single
    entry, as many columns as its vocabulary for a field answered with a set. The set's columns are
    padded out with the field's null, so a field's answer is the mean of the columns that are not its
    null -- and the null embedding itself when none of them are.

    Fields are pooled by summing, which is what keeps them independent: dropping the location leaves
    the anatomy's contribution untouched.
    """

    def __init__(self, prompt, width: int):
        super().__init__()
        self.width = width
        self.slots = tuple(field.slots for field in prompt.fields)
        self.offsets = prompt.offsets
        self.embedding = nn.Embedding(prompt.rows, width)

    def forward(self, prompt):
        if prompt.dim() == 1:
            return self.embedding(prompt)
        rows = self.embedding(prompt)
        pooled = None
        start = 0
        for slots, offset in zip(self.slots, self.offsets):
            columns = prompt[:, start:start + slots]
            vectors = rows[:, start:start + slots]
            answered = (columns != offset)[..., None]
            mean = (vectors * answered).sum(1) / answered.sum(1).clamp(min=1)
            field = torch.where(answered.any(1), mean, vectors[:, 0])
            pooled = field if pooled is None else pooled + field
            start += slots
        return pooled


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

    def __init__(self, prompt, width: int, feature_channels, gate_features: bool):
        super().__init__()
        self.encoder = CategoricalPromptEncoder(prompt, width)
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
