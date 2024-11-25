from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_dataset():
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3B")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk("data/processed")

if __name__ == "__main__":
    preprocess_dataset()
