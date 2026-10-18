import unittest

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from fm_adaptation.models import LayerNorm2d, UperNetDecoder

WIDTHS = (16, 32, 64, 128)


def _features(batch, size=32):
    return [torch.randn(batch, width, size // 2**level, size // 2**level)
            for level, width in enumerate(WIDTHS)]


class UperNetHeadNormTests(unittest.TestCase):
    def test_bn_head_keeps_batch_norm(self):
        decoder = UperNetDecoder(WIDTHS, 2)
        self.assertTrue(any(isinstance(m, _BatchNorm) for m in decoder.modules()))

    def test_ln_head_has_no_batch_norm_and_trains_on_one_image(self):
        decoder = UperNetDecoder(WIDTHS, 2, full_resolution=True, head_norm="ln").train()
        self.assertFalse(any(isinstance(m, _BatchNorm) for m in decoder.modules()))
        self.assertTrue(any(isinstance(m, LayerNorm2d) for m in decoder.head.modules()))
        logits = decoder(_features(1), (128, 128), images=torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(logits.shape), (1, 2, 128, 128))
        logits.mean().backward()


if __name__ == "__main__":
    unittest.main()
