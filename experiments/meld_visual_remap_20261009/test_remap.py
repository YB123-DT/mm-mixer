import unittest
from remap import signature, unique_match

class MappingTests(unittest.TestCase):
    def test_split_local_collision_requires_sentence_and_split(self):
        train=signature([0,1],['Hello','Goodbye'])
        dev=signature([0,1],['Different','Dialogue'])
        index={train:[0],dev:[1039,1153]}
        self.assertEqual(unique_match(train,index,{0,1039}),0)
        self.assertEqual(unique_match(dev,index,{0,1039}),1039)
        self.assertEqual(unique_match(dev,index,{1153}),1153)
    def test_ambiguity_is_rejected(self):
        sig=signature([0],['Hey'])
        with self.assertRaises(ValueError):unique_match(sig,{sig:[1,2]},{1,2})
    def test_order_and_utterance_ids_cannot_be_discarded(self):
        sig=signature([0,2],['One','Two'])
        self.assertNotEqual(sig,signature([0,1],['One','Two']))
        self.assertNotEqual(sig,signature([2,0],['Two','One']))
        with self.assertRaises(ValueError):unique_match(signature([0,1],['One','Two']),{sig:[9]},{9})
    def test_missing_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):signature([0,0],['A','B'])
        with self.assertRaises(ValueError):signature([0],['A','B'])
        with self.assertRaises(ValueError):unique_match(signature([0],['A']),{},set())
if __name__=='__main__':unittest.main()
