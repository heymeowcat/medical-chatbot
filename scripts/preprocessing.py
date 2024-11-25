from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_dataset():
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3B")

    def combine_conversations(examples):
        conversations = []
        for description, patient_msg, doctor_msg in zip(examples["Description"], examples["Patient"], examples["Doctor"]):
            conversation = f"Description: {description}\nPatient: {patient_msg}\nDoctor: {doctor_msg}"
            conversations.append(conversation)
        return {"text": conversations}

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    combined_dataset = dataset.map(combine_conversations, batched=True)
    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)
    
    if "train" in tokenized_dataset:
        tokenized_dataset["train"].save_to_disk("data/processed/train")

if __name__ == "__main__":
    preprocess_dataset()