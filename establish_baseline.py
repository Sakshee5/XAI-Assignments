from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the GPT-NEO model
model_name = "EleutherAI/gpt-neo-1.3B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define input text
input_text = "As she turned the corner, she noticed"

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=1,  # Prevents repetition of n-grams
    do_sample=True,          # Enable sampling to introduce randomness
    top_k=10,                # Top-k sampling to consider top-k tokens
    top_p=0.9,               # Top-p (nucleus) sampling to consider top-p tokens
    temperature=0.01         # Adjust temperature for randomness
)

# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)