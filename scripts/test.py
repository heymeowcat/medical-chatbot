from transformers import AutoModelForCausalLM, AutoTokenizer

original_model_name = "facebook/opt-125m"
fine_tuned_model_path = "models/opt-125m-model"

tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_model = AutoModelForCausalLM.from_pretrained(original_model_name)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)

prompts = [
    "Patient: I've been feeling dizzy and lightheaded. What should I do?",
    "Patient: What are the symptoms of diabetes?"
]
def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2, 
        repetition_penalty=1.2, 
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for prompt in prompts:
    print(f"Prompt: {prompt}")
    
    print("\n--- Original Model ---")
    original_response = generate_response(original_model, tokenizer, prompt)
    print(original_response)
    
    print("\n--- Fine-Tuned Model ---")
    fine_tuned_response = generate_response(fine_tuned_model, tokenizer, prompt)
    print(fine_tuned_response)
    
    print("\n" + "=" * 50 + "\n")
