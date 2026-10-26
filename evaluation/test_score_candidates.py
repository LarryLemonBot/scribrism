import unittest
import numpy as np

from score_candidates import features, fit_candidate, predict_candidate, grouped_folds, binary_metrics


class CandidateTests(unittest.TestCase):
    def test_features_order_and_floor(self):
        p = np.array([[[.1, .2, .3, .4], [.2, .3, .4, .1], [1, 0, 0, 0]]])
        expected = [np.log(2), np.log(3), np.log(4), np.log(1.5), np.log(2), np.log(.5)] + [np.log(1e-6)]*3
        np.testing.assert_allclose(features(p, 'R2')[0], expected)
        self.assertAlmostEqual(features(p, 'R1')[0, 0], np.log(4))

    def test_invalid_probabilities_block(self):
        for bad in [np.zeros((1, 3, 4)), np.ones((1, 3, 4)), np.full((1, 3, 4), np.nan)]:
            with self.assertRaises(ValueError): features(bad, 'R1')

    def test_folds_keep_all_related_versions_together(self):
        y = [0, 1]*30
        g = [f'writer{i//2}' for i in range(60)]
        f = grouped_folds(y, g)
        self.assertEqual(set(f), set(range(5)))
        for i in range(0, 60, 2): self.assertEqual(f[i], f[i+1])
        np.testing.assert_array_equal(f, grouped_folds(y, g))

    def test_group_failure_not_skipped(self):
        with self.assertRaises(ValueError): grouped_folds([0, 1], ['same', 'same'])
        with self.assertRaises(ValueError): grouped_folds([0]*10, [str(i) for i in range(10)])

    def test_fit_uses_training_only_and_serialized_prediction(self):
        x = np.repeat(np.array([[-2], [-1], [1], [2.]]), 9, axis=1)
        model = fit_candidate(x, [0, 0, 1, 1], 'R2')
        np.testing.assert_allclose(model['mean'], 0)
        before = model.copy()
        p = predict_candidate(model, x)
        predict_candidate(model, np.ones((2, 9))*1e6)
        self.assertEqual(before, model)
        self.assertTrue(np.all(p[:2] < .5) and np.all(p[2:] > .5))

    def test_negative_calibration_slope_rejected(self):
        with self.assertRaises(ValueError): fit_candidate([[2], [1], [-1], [-2]], [0, 0, 1, 1], 'R1')

    def test_failure_and_abstention_denominators(self):
        m = binary_metrics([0, 0, 1, 1, 1], ['AI', 'Human', 'Mixed', 'Uncertain', 'Error'])
        self.assertEqual(m['correct'], 2)
        self.assertEqual(m['coverage'], .6)
        self.assertEqual(m['failure_inclusive_binary_correctness'], .4)
        self.assertEqual(m['ai_recall'], 1/3)
        self.assertEqual((m['uncertain'], m['errors']), (1, 1))

    def test_absent_class_rates_are_undefined(self):
        m = binary_metrics([0], ['Human'])
        self.assertIsNone(m['ai_recall'])
        self.assertIsNone(m['balanced_coverage'])


if __name__ == '__main__': unittest.main()
