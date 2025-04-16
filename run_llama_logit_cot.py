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
import traceback

# Suppress INFO messages from transformers to avoid pad_token_id logging
import transformers
transformers.logging.set_verbosity_warning()

# Set up logging for the script
logging.basicConfig(filename='inference.log', level=logging.DEBUG,
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

def process_question(question, tokenizer, model, device, question_index):
    """Process a single question, return the raw output, answer based on logits, raw logits for A, B, C, D, and top 10 logits."""
    try:
        logging.debug(f"Processing question at index {question_index}: '{question[:50]}...'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at index {question_index}: must be a non-empty string, got '{question}'")

        # Prompt for chain of thought and answer
        prompt = (f"Question: ||{question}||\n"
                  "You must explain your reasoning step by step to determine the correct answer. "
                  "Break down the problem into clear steps, showing your thought process. "
                  "After explaining your reasoning, provide your final answer in the format {A}, {B}, {C}, or {D} (e.g., {B} for answer B). "
                  "The answer must be enclosed in curly braces and must be exactly one uppercase letter (A, B, C, or D). "
                  "Do not include any additional text after the answer.\n"
                  "Answer:")
        
        # Tokenize the prompt and generate the raw output
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        logging.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
        with torch.no_grad():
            # Generate the response with a reasonable max length
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the entire generated response (raw output)
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Raw model output: {raw_output}")

        # Compute raw logits for A, B, C, D and top 10 logits
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        logging.debug(f"Logits shape: {logits.shape}")
        last_token_logits = logits[0, -1]

        # Extract raw logits for A, B, C, D
        answer_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCD'}
        logging.debug(f"Answer token IDs: {answer_tokens}")
        for letter, token_id in answer_tokens.items():
            decoded = tokenizer.decode(token_id)
            logging.debug(f"Token ID {token_id} decodes to: '{decoded}'")

        answer_logits = {}
        for letter, token_id in answer_tokens.items():
            try:
                answer_logits[letter] = last_token_logits[token_id].item()
            except IndexError:
                answer_logits[letter] = 0.0

        logging.debug(f"Raw logits for A, B, C, D: {answer_logits}")

        # Extract top 10 logits (excluding A, B, C, D and their space-prefixed versions)
        answer_token_ids = set(answer_tokens.values())
        space_answer_tokens = {f" {letter}": tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in 'ABCD'}
        logging.debug(f"Space-prefixed token IDs: {space_answer_tokens}")
        exclude_token_ids = answer_token_ids.union(set(space_answer_tokens.values()))

        top_k_values, top_k_indices = torch.topk(last_token_logits, k=10 + len(exclude_token_ids))
        top_10_logits = {}
        for idx, value in zip(top_k_indices, top_k_values):
            token_id = idx.item()
            if token_id in exclude_token_ids:
                continue
            token = tokenizer.decode(token_id)
            top_10_logits[token] = value.item()
            if len(top_10_logits) >= 10:
                break

        logging.debug(f"Top 10 logits (excluding A, B, C, D and space-prefixed versions): {top_10_logits}")

        # Compute softmax probabilities to determine the answer (since we're not parsing the raw output)
        probs = torch.softmax(last_token_logits, dim=-1)
        answer_probs = {}
        for letter, token_id in answer_tokens.items():
            try:
                answer_probs[letter] = probs[token_id].item()
            except IndexError:
                answer_probs[letter] = 0.0

        total_prob = sum(answer_probs.values())
        if total_prob > 0:
            answer_probs = {letter: prob / total_prob for letter, prob in answer_probs.items()}
        else:
            answer_probs = {letter: 0.25 for letter in 'ABCD'}

        selected_answer = max(answer_probs, key=answer_probs.get)
        logging.debug(f"Answer based on probabilities: {selected_answer}, Probabilities (for selection only): {answer_probs}")

        if not is_valid_answer(selected_answer):
            logging.warning(f"Invalid answer at index {question_index}: '{selected_answer}' for question: '{question[:50]}...'")
            return "Error", answer_logits, raw_output, top_10_logits

        return selected_answer, answer_logits, raw_output, top_10_logits

    except Exception as e:
        logging.error(f"Error processing question at index {question_index} '{question[:50]}...': {str(e)}\n{traceback.format_exc()}")
        return "Error", {}, "Error in processing", {}

def main():
    args = parse_args()
    model_name = args.model_name
    prefix_type = args.prefix_type
    question_style = args.question_style
    input_filename = args.input_filename
    output_dir = args.output_dir
    max_retries = args.max_retries
    difficulty = args.difficulty
    full_question_col = args.full_question_col

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
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logging.info(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")

        print("Loading model...")
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        model = model.to(device)
        model.eval()

        print(f"Loading DataFrame from {input_filename}...")
        logging.info(f"Loading DataFrame from {input_filename}...")
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")
        logging.info(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")
        print(f"DataFrame index: {df.index}")
        logging.info(f"DataFrame index: {df.index}")
        print(f"DataFrame columns: {df.columns}")
        logging.info(f"DataFrame columns: {df.columns}")
        print("First few rows:\n", df.head())
        logging.info(f"First few rows:\n{df.head()}")

        if full_question_col not in df.columns:
            raise ValueError(f"Input DataFrame '{input_filename}' must contain a '{full_question_col}' column.")

        print(f"Validating '{full_question_col}' column contents...")
        logging.info(f"Validating '{full_question_col}' column contents...")
        invalid_questions = df[full_question_col].apply(lambda x: not isinstance(x, str) or not x.strip())
        if invalid_questions.any():
            invalid_indices = invalid_questions[invalid_questions].index.tolist()
            invalid_samples = df.loc[invalid_indices, full_question_col].head().to_dict()
            raise ValueError(f"Found {len(invalid_indices)} invalid questions in '{full_question_col}' column (must be non-empty strings). First few: {invalid_samples}")

        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "answer_logits" not in df.columns:
            df["answer_logits"] = None
        if "raw_output" not in df.columns:
            df["raw_output"] = None
        if "top_10_logits" not in df.columns:
            df["top_10_logits"] = None

        print("Testing with the first 5 questions...")
        logging.info("Testing with the first 5 questions...")
        for idx in df.index[:5]:
            question = df.at[idx, full_question_col]
            answer, logits, raw_out, top_10 = process_question(question, tokenizer, model, device, idx)
            print(f"Test question at index {idx}: Answer = {answer}, Logits = {logits}, Raw Output = {raw_out[:100]}..., Top 10 Logits = {top_10}")
            logging.info(f"Test question at index {idx}: Answer = {answer}, Logits = {logits}, Raw Output = {raw_out}, Top 10 Logits = {top_10}")
            torch.cuda.empty_cache()

        print("Processing all questions...")
        logging.info("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]):
                question = df.at[idx, full_question_col]
                answer, logits, raw_out, top_10 = process_question(question, tokenizer, model, device, idx)
                df.at[idx, "model_answer"] = answer
                df.at[idx, "answer_logits"] = logits
                df.at[idx, "raw_output"] = raw_out
                df.at[idx, "top_10_logits"] = top_10
                torch.cuda.empty_cache()

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
                question = df.at[idx, full_question_col]
                answer, logits, raw_out, top_10 = process_question(question, tokenizer, model, device, idx)
                df.at[idx, "model_answer"] = answer
                df.at[idx, "answer_logits"] = logits
                df.at[idx, "raw_output"] = raw_out
                df.at[idx, "top_10_logits"] = top_10
                torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        input_base = os.path.splitext(os.path.basename(input_filename))[0]
        dataset_name = input_base.split('_')[0]

        model_short_name = model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if question_style == "prefix_and_opinion":
            if prefix_type == "academic" and difficulty:
                output_filename = f"{output_dir}/{dataset_name}_{prefix_type}_opinion_{difficulty}_out_{model_short_name}_{timestamp_str}.pkl"
            else:
                output_filename = f"{output_dir}/{dataset_name}_{prefix_type}_opinion_out_{model_short_name}_{timestamp_str}.pkl"
        elif question_style == "opinion_only":
            output_filename = f"{output_dir}/{dataset_name}_opinion_only_out_{model_short_name}_{timestamp_str}.pkl"
        else:
            output_filename = f"{output_dir}/{dataset_name}_plain_out_{model_short_name}_{timestamp_str}.pkl"

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

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving to {output_filename}...")
        logging.info(f"Saving to {output_filename}...")
        df.to_pickle(output_filename)
        print(f"Completed and saved to {output_filename} with {len(df)} rows!")
        logging.info(f"Completed and saved to {output_filename} with {len(df)} rows!")

    except Exception as e:
        print(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        logging.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        raise

    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()