from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    model_name = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = load_from_disk("data/processed/train")


    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no", 
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        bf16=True,
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model("models/facebook/opt-125m")
    tokenizer.save_pretrained("models/facebook/opt-125m")

if __name__ == "__main__":
    train_model()