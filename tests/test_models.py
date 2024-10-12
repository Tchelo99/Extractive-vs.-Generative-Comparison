import unittest
from models.t5_model import get_t5_model
from models.bert_model import get_bert_model

class TestModels(unittest.TestCase):
    def test_t5_model(self):
        model = get_t5_model()
        self.assertIsNotNone(model)

    def test_bert_model(self):
        model = get_bert_model()
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
