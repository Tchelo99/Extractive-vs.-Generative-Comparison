import unittest
from data.preprocess_data import preprocess_function_t5, preprocess_function_bert
from transformers import T5Tokenizer, BertTokenizer

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.example = {
            "question": ["What is the capital of France?"],
            "context": ["Paris is the capital and most populous city of France."],
            "answers": {"text": ["Paris"], "answer_start": [0]}
        }

    def test_t5_preprocessing(self):
        processed = preprocess_function_t5(self.example, self.t5_tokenizer)
        self.assertIn("input_ids", processed)
        self.assertIn("attention_mask", processed)
        self.assertIn("labels", processed)

    def test_bert_preprocessing(self):
        processed = preprocess_function_bert(self.example, self.bert_tokenizer)
        self.assertIn("input_ids", processed)
        self.assertIn("attention_mask", processed)
        self.assertIn("start_positions", processed)
        self.assertIn("end_positions", processed)

if __name__ == "__main__":
    unittest.main()
