from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_t5_model(model_name="t5-small"):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def t5_inference(model, tokenizer, context, question):
    try:
        input_text = f"question: {question} context: {context}"
        inputs = tokenizer.encode(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )

        outputs = model.generate(
            inputs, max_length=64, num_beams=4, early_stopping=True
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer
    except Exception as e:
        print(f"Error in t5_inference: {str(e)}")
        return "An error occurred while processing the question."


if __name__ == "__main__":
    model, tokenizer = load_t5_model()
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    answer = t5_inference(model, tokenizer, context, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
