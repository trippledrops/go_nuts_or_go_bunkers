import os
import sys
import unittest
import torch
import torch.nn as nn

# Ensure samples/vit directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vit import MultiHeadSelfAttention


class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _init_weights(self, module: nn.Module):
        """Initializes model parameters to deterministic, non-trivial values."""
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

    def test_multi_head_self_attention_matches(self):
        """
        Tests that MultiHeadSelfAttention produces matching outputs with
        use_flash_attention=True and use_flash_attention=False on random inputs.
        """
        embed_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 32
        block_size = (8, 8)

        # 1. Instantiate the module and initialize weights
        model = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            use_flash_attention=False,
            flash_attn_block_size=block_size,
        )
        self._init_weights(model)
        model.eval()

        # 2. Generate random input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)

        # 3. Perform inference with and without flash attention
        with torch.no_grad():
            # Standard attention
            model.use_flash_attention = False
            out_standard = model(x)

            # Flash attention
            model.use_flash_attention = True
            out_flash = model(x)

        # 4. Validate output shapes and numerical closeness
        self.assertEqual(
            out_flash.shape,
            out_standard.shape,
            f"Shape mismatch: flash={out_flash.shape} vs standard={out_standard.shape}",
        )
        torch.testing.assert_close(
            out_flash,
            out_standard,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda msg: f"Outputs do not match between flash and standard attention: {msg}",
        )

    def test_infer_flash_attention_direct(self):
        """
        Directly tests infer_flash_attention(q, k, v) against the standard
        attention equation: softmax(Q @ K.T * scale) @ V.
        """
        embed_dim = 64
        num_heads = 4
        head_dim = embed_dim // num_heads
        batch_size = 2
        seq_len = 32
        block_size = (8, 8)

        model = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            flash_attn_block_size=block_size,
        )
        model.eval()

        # Generate random Q, K, V
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        with torch.no_grad():
            # Standard attention formula
            scale = head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            expected_out = attn @ v

            # Method under test
            flash_out = model.infer_flash_attention(q, k, v)

        self.assertEqual(
            flash_out.shape,
            expected_out.shape,
            f"Shape mismatch: flash={flash_out.shape} vs expected={expected_out.shape}",
        )
        torch.testing.assert_close(
            flash_out,
            expected_out,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda msg: f"Direct flash attention output does not match expected output: {msg}",
        )


if __name__ == "__main__":
    unittest.main()
