import unittest
from inference.t5_inference import load_t5_model, t5_inference
from inference.bert_inference import load_bert_model, bert_inference

class TestInference(unittest.TestCase):
    def setUp(self):
        self.t5_model, self.t5_tokenizer = load_t5_model()
        self.bert_model, self.bert_tokenizer = load_bert_model()
        self.question = "What is the capital of France?"
        self.context = "Paris is the capital and most populous city of France."

    def test_t5_inference(self):
        answer = t5_inference(self.question, self.context, self.t5_model, self.t5_tokenizer)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_bert_inference(self):
        answer = bert_inference(self.question, self.context, self.bert_model, self.bert_tokenizer)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

if __name__ == "__main__":
    unittest.main()
