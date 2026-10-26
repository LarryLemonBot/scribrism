"""Synthetic tests for the frozen region mechanism, without real scores."""
import inspect
import math
import unittest

from region_candidate import ordered_step_region, planned_windows


def make_windows(n=2048, score_fn=None):
    score_fn = score_fn or (lambda s, e: [0.9, 0.1, 0, 0] if (s+e)/2 < 1024 else [0.1, 0.9, 0, 0])
    return [(s, e, score_fn(s, e)) for s, e in planned_windows(n)]


class RegionCandidateTests(unittest.TestCase):
    def test_blind_step_and_reversed_step(self):
        forward = ordered_step_region(2048, make_windows())
        reverse = ordered_step_region(2048, [(s, e, [p[1], p[0], p[2], p[3]]) for s, e, p in make_windows()])
        self.assertTrue(forward['flag'])
        self.assertTrue(reverse['flag'])
        self.assertEqual((forward['order'], reverse['order']), ('HA', 'AH'))
        self.assertEqual(forward['boundary'], reverse['boundary'])
        self.assertLessEqual(abs(forward['boundary']-1024), 128)

    def test_lower_confidence_coherent_step_can_fire_outside_old_D(self):
        wins = make_windows(score_fn=lambda s,e: [0.65,0.35,0,0] if (s+e)/2 < 1024 else [0.35,0.65,0,0])
        self.assertTrue(ordered_step_region(2048, wins)['flag'])
        self.assertFalse(any(max(p[0],p[1]+p[3]) >= .9 for _,_,p in wins))

    def test_constant_profile_is_valid_negative(self):
        result = ordered_step_region(2048, make_windows(score_fn=lambda s,e: [.9,.1,0,0]))
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['flag'])
        self.assertIn('near_constant_profile', result['grids'][0]['reason_codes'])

    def test_linear_drift_rejected(self):
        def linear(s,e):
            q = -0.9 + 1.8 * (((s+e)/2-128)/1792)
            return [(1-q)/2, (1+q)/2, 0, 0]
        result = ordered_step_region(2048, make_windows(score_fn=linear))
        self.assertFalse(result['flag'])
        self.assertTrue(all('linear_trend_not_rejected' in g['reason_codes'] for g in result['grids']))

    def test_scattered_high_confidence_regions_rejected(self):
        wins = make_windows(score_fn=lambda s,e: [.99,.01,0,0] if (s//256)%2 == 0 else [.01,.99,0,0])
        self.assertFalse(ordered_step_region(2048, wins)['flag'])

    def test_edited_class_never_counts_as_AI_anchor(self):
        wins = make_windows(score_fn=lambda s,e: [.9,.1,0,0] if (s+e)/2 < 1024 else [.1,0,.9,0])
        result = ordered_step_region(2048, wins)
        self.assertFalse(result['flag'])
        self.assertTrue(all('opposed_segment_anchors_absent' in g['reason_codes'] for g in result['grids']))

    def test_humanized_is_in_AI_anchor(self):
        wins = make_windows(score_fn=lambda s,e: [.9,.1,0,0] if (s+e)/2 < 1024 else [.1,0,0,.9])
        self.assertTrue(ordered_step_region(2048, wins)['flag'])

    def test_both_grids_require_four_full_windows(self):
        self.assertEqual(ordered_step_region(1024, make_windows(1024))['status'], 'insufficient_resolution')
        out = ordered_step_region(1152, make_windows(1152))
        self.assertEqual([g['n_windows'] for g in out['grids']], [4,4])
        self.assertEqual(out['status'], 'complete')
        self.assertIsNone(ordered_step_region(386, make_windows(386))['flag'])

    def test_invalid_and_missing_inputs_never_become_negative(self):
        wins = make_windows()
        for case in (wins[:-1], wins+[wins[0]], [(0,256,[math.nan,.1,0,0])]+wins[1:],
                     [(0,256,[.1,.1,.1,.1])]+wins[1:], [(True,256,[.9,.1,0,0])]+wins[1:]):
            with self.subTest(case=repr(case[:1])):
                out = ordered_step_region(2048, case)
                self.assertEqual(out['status'], 'invalid_input')
                self.assertIsNone(out['flag'])
        for n in (True, 0, -1, 2048.0, None):
            self.assertEqual(ordered_step_region(n, wins)['status'], 'invalid_input')

    def test_irregular_tail_validated_but_not_analyzed(self):
        n = 2051
        wins = make_windows(n)
        self.assertEqual(wins[-1][0], 1795)
        out = ordered_step_region(n, wins)
        self.assertNotIn(1795, [s for g in out['grids'] for s in g['window_starts']])
        self.assertEqual(ordered_step_region(n, wins[:-1])['status'], 'invalid_input')

    def test_grid_order_disagreement_is_negative(self):
        def p(s,e):
            human = ((s+e)/2 < 1024) == (s%256 == 0)
            return [.9,.1,0,0] if human else [.1,.9,0,0]
        out = ordered_step_region(2048, make_windows(score_fn=p))
        self.assertTrue(all(g['flag'] for g in out['grids']))
        self.assertFalse(out['flag'])
        self.assertEqual(out['reason_codes'], ['grid_order_disagreement'])

    def test_grid_boundary_disagreement_is_negative(self):
        def p(s,e):
            boundary = 512 if s%256 == 0 else 1408
            return [.9,.1,0,0] if (s+e)/2 < boundary else [.1,.9,0,0]
        out = ordered_step_region(2048, make_windows(score_fn=p))
        self.assertTrue(all(g['flag'] for g in out['grids']))
        self.assertEqual(out['reason_codes'], ['grid_boundary_disagreement'])

    def test_window_order_has_no_effect_and_input_is_not_mutated(self):
        wins = make_windows()
        self.assertEqual(ordered_step_region(2048,wins),ordered_step_region(2048,list(reversed(wins))))
        self.assertEqual(wins, make_windows())

    def test_truth_metadata_has_no_inference_argument(self):
        self.assertEqual(list(inspect.signature(ordered_step_region).parameters), ['n_tokens','windows'])
        with self.assertRaises(TypeError):
            ordered_step_region(2048, make_windows(), order='HA')


if __name__ == '__main__':
    unittest.main()
