import torch
import config
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get Hugging Face token from environment variable
hf_token = config.HF_TOKEN
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it with your Hugging Face token.")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"

try:
    # Load tokenizer with increased timeout and token
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, timeout=30)

    # Load model with increased timeout and token
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, timeout=30)

    # Move model to GPU
    model = model.to(device)
    model.eval()

    # Example input text
    input_text = "Hello, how can I assist you today?"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate output
    print("Generating output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your internet connection or try again later.")

# Clear GPU memory
torch.cuda.empty_cache()