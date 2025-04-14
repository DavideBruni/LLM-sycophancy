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
    parser.add_argument("--prefix_type", type=str, default="", 
                        choices=["academic", "behavior", ""], 
                        help="Type of prefix used (e.g., 'academic', 'behavior'), or '' for no prefix.")
    parser.add_argument("--question_style", type=str, default="plain", 
                        choices=["prefix_and_opinion", "opinion_only", "plain"], 
                        help="Style of the questions: 'prefix_and_opinion' (prefix + opinion), "
                             "'opinion_only' (just opinion), or 'plain' (no prefix or opinion).")
    parser.add_argument("--input_filename", type=str, default="output/mmlupro/mmlupro_plain.pkl", 
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--output_dir", type=str, default="output/mmlupro", help="Output directory")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    parser.add_argument("--difficulty", type=str, default="", 
                        choices=["", "beginner", "intermediate", "expert"], 
                        help="Difficulty level for academic_opinion style (beginner, intermediate, expert). "
                             "Only applies when question_style='prefix_and_opinion' and prefix_type='academic'.")
    parser.add_argument("--full_question_col", type=str, default="full_question", 
                        help="Name of the column in the input DataFrame containing the full question text")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def process_question(question, tokenizer, model, device):
    """Process a single question, return the model's answer and probabilities for A, B, C, D."""
    try:
        prompt = f"Question: ||{question}||\nRespond with exactly one uppercase letter (A, B, C, D, etc.) and nothing else.\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        last_token_logits = logits[0, -1]
        probs = torch.softmax(last_token_logits, dim=-1)

        answer_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCD'}
        answer_probs = {}
        for letter, token_id in answer_tokens.items():
            try:
                answer_probs[letter] = probs[token_id].item()
            except IndexError:
                answer_probs[letter] = 0.0

        max_prob_letter = max(answer_probs, key=answer_probs.get)
        if not is_valid_answer(max_prob_letter):
            logging.warning(f"Invalid answer predicted: '{max_prob_letter}' for question: '{question[:50]}...'")
            return "Error", answer_probs

        return max_prob_letter, answer_probs

    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return "Error", {}

def main():
    args = parse_args()
    model_name = args.model_name
    prefix_type = args.prefix_type
    question_style = args.question_style
    input_filename = args.input_filename
    output_dir = args.output_dir
    max_retries = args.max_retries
    difficulty = args.difficulty
    full_question_col = args.full_question_col  # New argument for column name

    # Validation: Ensure difficulty is only specified for academic_opinion
    if difficulty and not (question_style == "prefix_and_opinion" and prefix_type == "academic"):
        raise ValueError("The --difficulty argument is only applicable when question_style='prefix_and_opinion' "
                         "and prefix_type='academic'.")

    if question_style == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_style, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")

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

        # Check if the specified full_question column exists
        if full_question_col not in df.columns:
            raise ValueError(f"Input DataFrame '{input_filename}' must contain a '{full_question_col}' column.")

        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "answer_probs" not in df.columns:
            df["answer_probs"] = None

        # Process each question using the specified column
        questions = df[full_question_col].tolist()
        for i, question in tqdm(enumerate(questions), total=len(questions), desc="Initial processing"):
            if not is_valid_answer(df.at[i, "model_answer"]):
                answer, probs = process_question(question, tokenizer, model, device)
                df.at[i, "model_answer"] = answer
                df.at[i, "answer_probs"] = probs

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
                question = df.at[idx, full_question_col]  # Use the specified column
                answer, probs = process_question(question, tokenizer, model, device)
                df.at[idx, "model_answer"] = answer
                df.at[idx, "answer_probs"] = probs

            retry_count += 1
            time.sleep(1)

        # Extract dataset name from input filename (e.g., 'mmlu' from 'mmlu_plain.pkl')
        input_base = os.path.splitext(os.path.basename(input_filename))[0]  # e.g., 'mmlu_plain'
        dataset_name = input_base.split('_')[0]  # e.g., 'mmlu'

        # Define output filename using dataset_name
        model_short_name = model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if question_style == "prefix_and_opinion":
            if prefix_type == "academic" and difficulty:
                output_filename = f"{output_dir}/{dataset_name}_{prefix_type}_opinion_{difficulty}_out_{model_short_name}_{timestamp_str}.pkl"
            else:
                output_filename = f"{output_dir}/{dataset_name}_{prefix_type}_opinion_out_{model_short_name}_{timestamp_str}.pkl"
        elif question_style == "opinion_only":
            output_filename = f"{output_dir}/{dataset_name}_opinion_only_out_{model_short_name}_{timestamp_str}.pkl"
        else:  # plain
            output_filename = f"{output_dir}/{dataset_name}_plain_out_{model_short_name}_{timestamp_str}.pkl"

        # Check for invalid answers and save regardless
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