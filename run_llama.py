import torch
import pandas as pd
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

input_filename = "output/mmlupro/mmlupro_with_academic.pkl"
output_filename = "output/mmlupro/mmlupro_with_academic_out.pkl"
model_name = "meta-llama/Llama-3.2-1B"

# Get Hugging Face token from environment variable
hf_token = config.HF_TOKEN
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it with your Hugging Face token.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    model = model.to(device)
    model.eval()

    # ===== Load your dataframe =====
    df = pd.read_pickle(input_filename)
    print(f"Loaded dataframe with {len(df)} entries.")

    questions = df["full_question"].tolist()

    llama_answers = []

    # ===== Inference loop =====
    for question in tqdm(questions, desc="Processing questions"):
        try:
            # Create a strong prompt to restrict model output
            prompt = f"Output only a letter such as A, B, C, D, etc.:\n\n{question}\n\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,  # Only generate one token
                    temperature=0.0,   # Very deterministic
                    do_sample=False,   # No random sampling
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_token = outputs[0][inputs['input_ids'].shape[-1]:]  # Only the new token(s)
            generated_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()

            # Save the generated single letter
            llama_answers.append(generated_text)

        except Exception as e:
            print(f"Error processing question: {e}")
            llama_answers.append("Error")

    # ===== Save the new dataframe =====
    df["llama_1b_answer"] = llama_answers
    df.to_pickle(output_filename)

    print("Completed and saved!")

except Exception as e:
    print(f"An error occurred during setup: {str(e)}")
    print("Please check your model, tokenizer, or environment.")

torch.cuda.empty_cache()