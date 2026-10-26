import unittest

import torch

try:
    from .masked_pooling import masked_mean
except ImportError:
    from masked_pooling import masked_mean


class MaskedMeanTests(unittest.TestCase):
    def test_padding_values_do_not_change_mean_or_mutate_inputs(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [100., -200.]]])
        mask = torch.tensor([[1, 1, 0]])
        before, mask_before = hidden.clone(), mask.clone()
        actual = masked_mean(hidden, mask)
        torch.testing.assert_close(actual, torch.tensor([[2., 3.]]), rtol=0, atol=0)
        torch.testing.assert_close(hidden, before, rtol=0, atol=0)
        torch.testing.assert_close(mask, mask_before, rtol=0, atol=0)

    def test_layered_output_preserves_batch_then_layer_order(self):
        hidden = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 2)
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]])
        actual = masked_mean(hidden, mask)
        expected = torch.stack([torch.stack([
            hidden[layer, batch, mask[batch].bool()].mean(dim=0)
            for layer in range(2)]) for batch in range(3)])
        self.assertEqual(actual.shape, (3, 2, 2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_no_padding_matches_plain_mean_for_supported_float_types(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            hidden = torch.tensor([[[2., 4.], [4., 8.]]], dtype=dtype)
            actual = masked_mean(hidden, torch.ones(1, 2, dtype=torch.bool))
            self.assertEqual(actual.dtype, dtype)
            torch.testing.assert_close(actual, hidden.mean(dim=1), rtol=0, atol=0)

    def test_noncontiguous_tensor_is_supported(self):
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)[..., ::2]
        mask = torch.tensor([[1., 0., 1.]])
        self.assertFalse(hidden.is_contiguous())
        torch.testing.assert_close(masked_mean(hidden, mask), hidden[:, [0, 2]].mean(dim=1))

    def test_nonfinite_padding_is_ignored_but_valid_states_are_rejected(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            hidden = torch.tensor([[[2., 4.], [invalid, invalid]]])
            torch.testing.assert_close(masked_mean(hidden, torch.tensor([[1, 0]])), hidden[:, 0])
            with self.assertRaisesRegex(ValueError, 'Nonfinite valid'):
                masked_mean(hidden, torch.tensor([[1, 1]]))

    def test_empty_mask_row_is_rejected_even_when_other_rows_are_valid(self):
        with self.assertRaisesRegex(ValueError, 'at least one valid token'):
            masked_mean(torch.ones(2, 2, 3), torch.tensor([[1, 0], [0, 0]]))

    def test_nonbinary_and_nonfinite_masks_are_rejected(self):
        for invalid in (-1., .5, 2., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'only zero or one'):
                masked_mean(torch.ones(1, 2, 3), torch.tensor([[1., invalid]]))

    def test_invalid_shapes_types_and_empty_dimensions_are_rejected(self):
        for hidden, mask in [
            (torch.ones(2, 3), torch.ones(2, 3)),
            (torch.ones(1, 2, 3), torch.ones(1, 3)),
            (torch.ones(1, 2, 3), torch.ones(2)),
            (torch.ones(0, 2, 3), torch.ones(0, 2)),
            (torch.ones(1, 0, 3), torch.ones(1, 0)),
            (torch.ones(1, 2, 0), torch.ones(1, 2)),
            (torch.ones(0, 1, 2, 3), torch.ones(1, 2)),
            (torch.ones(1, 2, 3, dtype=torch.int64), torch.ones(1, 2)),
            (torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.complex64)),
            ([], torch.ones(1, 2)),
        ]:
            with self.subTest(hidden=getattr(hidden, 'shape', None), mask=mask.shape):
                with self.assertRaises(ValueError):
                    masked_mean(hidden, mask)

    def test_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite pooled'):
            masked_mean(torch.full((1, 2, 1), 60000., dtype=torch.float16), torch.ones(1, 2))

    def test_gradient_ignores_padding_without_inplace_mutation(self):
        hidden = torch.ones(1, 3, 2, requires_grad=True)
        masked_mean(hidden, torch.tensor([[1, 0, 1]])).sum().backward()
        torch.testing.assert_close(hidden.grad, torch.tensor([[[.5, .5], [0., 0.], [.5, .5]]]))


if __name__ == '__main__':
    unittest.main()
