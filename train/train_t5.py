from models.t5_model import get_t5_model
import datasets
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)


def train_t5():
    # Load preprocessed dataset
    dataset = datasets.load_from_disk("data/processed_squad_t5")

    # Load model and tokenizer
    model = get_t5_model()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Train the model
    trainer.train()

    # Save the model
    model_save_path = os.path.join(project_root, "models/t5_qa")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_t5()
    print("T5 model trained successfully.")
