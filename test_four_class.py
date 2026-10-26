import unittest
import four_class as engine
class CorrectedMappingTests(unittest.TestCase):
    def setUp(self):self.c={'minimum_score':.6,'region_confidence':.8,'mixed_min_fraction':.1,'region_override':True}
    def test_ai_paraphrase_is_not_human_contribution(self):
        self.assertEqual(engine.classify([[0,.1,0,.9]]*10,self.c)['label'],'ai')
    def test_human_text_edited_with_ai_is_mixed_candidate(self):
        self.assertEqual(engine.classify([[0,0,1,0]]*10,self.c)['label'],'mixed')
    def test_ai_region_combines_raw_and_rewritten_ai(self):
        self.assertEqual(engine.classify([[1,0,0,0]]*10+[[0,.4,0,.6]]*10,self.c)['label'],'mixed')
    def test_humanized_only_is_not_mixed_region(self):
        self.assertEqual(engine.classify([[0,0,0,1]]*10,self.c)['label'],'ai')
if __name__=='__main__':unittest.main()
