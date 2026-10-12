import unittest

import torch
import torch.nn.functional as F

from fm_adaptation.export_onnx import FixedAdaptiveAvgPool2d


class FixedPoolingTests(unittest.TestCase):
        """The matrix form adaptive pooling is exported as."""

        def test_matches_adaptive_pooling_when_the_output_does_not_divide_the_input(self):
            x = torch.rand(2, 5, 16, 12)
            for size in ((1, 1), (2, 2), (3, 3), (6, 6), (3, 5)):
                fixed = FixedAdaptiveAvgPool2d((16, 12), size)
                self.assertTrue(torch.allclose(fixed(x), F.adaptive_avg_pool2d(x, size), atol=1e-6), size)


if __name__ == "__main__":
    unittest.main()
