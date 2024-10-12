import torch
from transformers import BertForQuestionAnswering, BertTokenizer


def load_bert_model(model_path="models/bert_qa"):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def bert_inference(model, tokenizer, context, question):
    try:
        inputs = tokenizer(
            question, context, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])

        return answer
    except Exception as e:
        print(f"Error in bert_inference: {str(e)}")
        return "An error occurred while processing the question."


if __name__ == "__main__":
    model, tokenizer = load_bert_model()
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    answer = bert_inference(model, tokenizer, context, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
