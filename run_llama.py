import torch
import pandas as pd
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import logging
import os

# Set up logging
logging.basicConfig(filename='inference.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference on a dataset with pre-constructed questions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name for inference")
    parser.add_argument("--prefix_type", type=str, default="", choices=["academic", "behavior", ""], help="Type of prefix used in the input file (for naming purposes). Use '' if no prefix.")
    parser.add_argument("--input_filename", type=str, default="output/mmlupro/mmlupro_academic.pkl", help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--output_dir", type=str, default="output/mmlupro", help="Output directory")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def process_question(question, tokenizer, model, device):
    """Process a single question and return the model's answer."""
    try:
        prompt = f"Output only a single uppercase letter (A, B, C, D, etc.) and nothing else:\n\n{question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_token = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()

        if not is_valid_answer(generated_text):
            logging.warning(f"Invalid output for question: '{question[:50]}...'. Generated: '{generated_text}'")

        return generated_text
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return "Error"

def main():
    args = parse_args()
    model_name = args.model_name
    prefix_type = args.prefix_type
    input_filename = args.input_filename
    output_dir = args.output_dir
    max_retries = 3

    hf_token = config.HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        model = model.to(device)
        model.eval()

        # Load the pre-constructed DataFrame
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")
        logging.info(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")

        # Ensure the DataFrame has a 'full_question' column
        if "full_question" not in df.columns:
            raise ValueError(f"Input DataFrame '{input_filename}' must contain a 'full_question' column.")

        # Ensure the DataFrame has a column for LLaMA answers
        if "model_answer" not in df.columns:
            df["model_answer"] = None

        # Process each question using the LLaMA model
        questions = df["full_question"].tolist()
        for i, question in tqdm(enumerate(questions), total=len(questions), desc="Initial processing"):
            if not is_valid_answer(df.at[i, "model_answer"]):
                answer = process_question(question, tokenizer, model, device)
                df.at[i, "model_answer"] = answer

        # Retry loop for invalid answers
        retry_count = 0
        while retry_count < max_retries:
            invalid_indices = df.index[
                df["model_answer"].isna() |
                (df["model_answer"] == "") |
                (df["model_answer"] == "Error") |
                (~df["model_answer"].apply(is_valid_answer))
            ].tolist()

            if not invalid_indices:
                print("All entries have valid answers!")
                logging.info("All entries have valid answers!")
                break

            print(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            logging.info(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            for idx in tqdm(invalid_indices, desc=f"Retry {retry_count + 1}"):
                question = df.at[idx, "full_question"]
                answer = process_question(question, tokenizer, model, device)
                df.at[idx, "model_answer"] = answer

            retry_count += 1
            time.sleep(1)

        # Check for remaining invalid answers
        invalid_count = len(df[
            df["model_answer"].isna() |
            (df["model_answer"] == "") |
            (df["model_answer"] == "Error") |
            (~df["model_answer"].apply(is_valid_answer))
        ])
        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries still have invalid answers after {max_retries} retries.")
            logging.warning(f"{invalid_count} entries still have invalid answers after {max_retries} retries.")
        else:
            print("All entries successfully populated with valid answers!")
            logging.info("All entries successfully populated with valid answers!")

        # Save the output DataFrame
        model_short_name = model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_identifier = f"_{prefix_type}_out" if prefix_type else "_no_prefix_out"
        output_filename = f"{output_dir}/mmlupro{prefix_identifier}_{model_short_name}_{timestamp_str}.pkl"
        if prefix_type == "":
            output_filename = f"{output_dir}/mmlupro_no_prefix_{model_short_name}_{timestamp_str}.pkl"


        df.to_pickle(output_filename)
        print(f"Completed and saved to {output_filename} with {len(df)} rows!")
        logging.info(f"Completed and saved to {output_filename} with {len(df)} rows!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")
        print("Please check your model, tokenizer, or environment.")

    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()