from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_dataset():
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    def combine_conversations(examples):
        conversations = []
        for description, patient_msg, doctor_msg in zip(examples["Description"], examples["Patient"], examples["Doctor"]):
            conversation = (
                f"Description: {description}\n"
                f"Patient: {patient_msg}\n"
                f"Doctor: {doctor_msg}"
            )
            conversations.append(conversation)
        return {"text": conversations}

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=204
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    combined_dataset = dataset.map(combine_conversations, batched=True)
    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

    tokenized_dataset["train"].save_to_disk("data/processed/train")

if __name__ == "__main__":
    preprocess_dataset()
