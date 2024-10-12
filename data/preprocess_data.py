import datasets
from transformers import T5Tokenizer, BertTokenizerFast


def preprocess_function_t5(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = [f"question: {q}  context: {c}" for q, c in zip(questions, contexts)]

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )

    # Extract the first answer text for each example (since answers are lists)
    answers = [
        ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]
    ]

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            answers, max_length=64, truncation=True, padding="max_length"
        )

    # Set the labels in model_inputs
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def preprocess_function_bert(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def preprocess_squad():

    squad = datasets.load_from_disk("data/squad")

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Preprocess for T5
    t5_dataset = squad.map(
        lambda examples: preprocess_function_t5(examples, t5_tokenizer),
        batched=True,
        remove_columns=squad["train"].column_names,
    )

    # Preprocess for BERT
    bert_dataset = squad.map(
        lambda examples: preprocess_function_bert(
            examples, bert_tokenizer
        ),  # Use the new tokenizer
        batched=True,
        remove_columns=squad["train"].column_names,
    )

    # Save preprocessed datasets
    t5_dataset.save_to_disk("data/processed_squad_t5")
    bert_dataset.save_to_disk("data/processed_squad_bert")


if __name__ == "__main__":
    preprocess_squad()
    print("SQuAD dataset preprocessed for T5 and BERT successfully.")
