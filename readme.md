# Question Answering System using T5 and BERT

This project implements a question-answering system using T5 (Text-to-Text Transfer Transformer) and BERT (Bidirectional Encoder Representations from Transformers) models. Users can ask questions based on provided texts, and the model returns precise answers.

## Features

- Fine-tuned T5 and BERT models on the SQuAD dataset
- Flask web application for easy interaction
- Comparison between extractive QA (BERT) and generative QA (T5)

## Installation

1. Clone this repository:
   ```
   https://github.com/Tchelo99/QA-Extractive-vs-Generative-Comparison.git
   cd QA-Extractive-vs-Generative-Comparison
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download and preprocess the SQuAD dataset:
   ```
   python data/download_squad.py
   python data/preprocess_data.py
   ```

2. Train the models:
   ```
   python train/train_t5.py
   python train/train_bert.py
   ```

3. Run the Flask application:
   ```
   python run.py
   ```

4. Open your web browser and navigate to `http://localhost:5000` to use the question-answering system.

## Testing

Run the tests using pytest:
```
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
