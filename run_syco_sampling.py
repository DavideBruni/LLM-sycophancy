import torch
import pandas as pd
import config 
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import logging
import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast

# Set up logging
logging.basicConfig(filename='sampling.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with sampling on a dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name for inference")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu"], help="Dataset to use (currently only 'mmlu' supported)")
    parser.add_argument("--prefix_type", type=str, default="", 
                        choices=["academic", "behavior", ""], 
                        help="Type of prefix used (e.g., 'academic', 'behavior').")
    parser.add_argument("--academic_level", type=str, default="", 
                        choices=["beginner", "intermediate", "advanced", ""], 
                        help="Academic level for academic prefix (beginner, intermediate, advanced). "
                             "Only applies when prefix_type='academic'.")
    parser.add_argument("--prefix_subtype", type=str, default="", 
                        choices=["original", "mixing_subject", "third_pov", ""], 
                        help="Subtype of prefix (original, mixing_subject, third_pov).")
    parser.add_argument("--question_type", type=str, default="plain", 
                        choices=["prefix_and_opinion", "opinion_only", "plain"], 
                        help="Type of the questions: 'prefix_and_opinion' (prefix + opinion), "
                             "'opinion_only' (just opinion), or 'plain' (no prefix or opinion).")
    parser.add_argument("--input_filename", type=str, default="output_sampling/mmlu/mmlu_plain.pkl", 
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--full_question_column", type=str, default="full_question", 
                        help="Name of the column containing the full question text in the input DataFrame")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    parser.add_argument("--sampling", type=int, default=1, 
                        help="Number of samples to generate for each question (default: 1 for regular inference)")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Directory to cache the model files (defaults to ~/.cache/huggingface/hub)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing questions (default: 8)")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def process_batch(questions, tokenizer, model, sampling, question_indices, device):
    try:
        # Prepare prompts
        prompts = [f"Question: ||{q}||\nRespond with exactly one uppercase letter (A, B, C, D) and nothing else.\nAnswer:" for q in questions]
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Process multiple samples
        batch_answers = [[] for _ in range(len(questions))]
        answer_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCD'}
        space_answer_tokens = {letter: tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in 'ABCD'}

        for _ in range(sampling):
            try:
                with autocast():  # Mixed precision for faster inference
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]  # Logits for the last token

                # Extract logits for A, B, C, D
                for i in range(len(questions)):
                    answer_logits = {}
                    for letter in 'ABCD':
                        token_id = answer_tokens[letter]
                        space_token_id = space_answer_tokens[letter]
                        logit = max(
                            logits[i, token_id].item(),
                            logits[i, space_token_id].item()
                        )
                        answer_logits[letter] = logit

                    # Convert logits to probabilities
                    logits_tensor = torch.tensor([answer_logits[letter] for letter in 'ABCD']).to(device)
                    logits_tensor = logits_tensor - logits_tensor.max()  # Numerical stabilization
                    if sampling > 1:
                        probs = torch.softmax(logits_tensor / 0.7, dim=-1)  # Temperature for sampling
                    else:
                        probs = torch.softmax(logits_tensor, dim=-1)  # No temperature for deterministic
                    probs = torch.clamp(probs, min=1e-9, max=1.0 - 1e-9)  # Prevent invalid probabilities
                    probs = probs / probs.sum()  # Re-normalize

                    if sampling > 1:
                        next_token_idx = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token_idx = torch.argmax(probs).item()
                    generated_text = 'ABCD'[next_token_idx]

                    if generated_text and is_valid_answer(generated_text):
                        result = generated_text
                    else:
                        logging.warning(f"Invalid output for question {question_indices[i]}: '{questions[i][:50]}...'. Generated: '{generated_text}'")
                        result = ""

                    batch_answers[i].append(result)

            except Exception as e:
                logging.error(f"Error in sampling iteration for batch at indices {question_indices}: {str(e)}")
                for i in range(len(questions)):
                    batch_answers[i].append("Error")

        return batch_answers

    except Exception as e:
        logging.error(f"Error processing batch at indices {question_indices}: {str(e)}\n{traceback.format_exc()}")
        return [["Error"] * sampling for _ in range(len(questions))]

def main():
    args = parse_args()
    model_name = args.model_name
    dataset = args.dataset
    prefix_type = args.prefix_type
    academic_level = args.academic_level
    prefix_subtype = args.prefix_subtype
    question_type = args.question_type
    input_filename = args.input_filename
    full_question_column = args.full_question_column
    max_retries = args.max_retries
    sampling = args.sampling
    cache_dir = args.cache_dir
    batch_size = args.batch_size

    # Validation
    if academic_level and prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")
    if question_type == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")
    if sampling < 1:
        raise ValueError("Sampling must be a positive integer (>= 1).")
    if batch_size < 1:
        raise ValueError("Batch size must be a positive integer (>= 1).")

    hf_token = config.HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN not found in config module.")

    try:
        print("Loading tokenizer...")
        logging.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                #token=hf_token, # for new cuda version
                use_auth_token=hf_token, # for old cuda version
                trust_remote_code=True,
                cache_dir=cache_dir,
                padding_side='left',
                use_fast=False
            )
            tokenizer.pad_token = tokenizer.eos_token
        except ValueError as e:
            logging.error(f"Failed to load tokenizer for model {model_name}: {str(e)}")
            print(f"Error: Unrecognized model {model_name}. Please check the model name or update the Transformers library.")
            print("Suggestions:")
            print("- Verify the model name (e.g., 'Qwen/Qwen2.5-7B-Instruct' should be correct).")
            print("- Run 'pip install --upgrade transformers' to ensure you have the latest version.")
            print(f"- Clear the cache in {cache_dir or '~/.cache/huggingface/hub'} and try again.")
            raise

        print("Loading Transformers model...")
        logging.info("Loading Transformers model...")
        try:
            configuration = AutoConfig.from_pretrained(
                model_name,
                #token=hf_token,   # for new cuda version
                use_auth_token=hf_token,  # for old cuda version
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        except ValueError as e:
            logging.error(f"Failed to load configuration for model {model_name}: {str(e)}")
            print(f"Error: Unrecognized model {model_name}. Please check the model name or update the Transformers library.")
            print("Suggestions:")
            print("- Verify the model name (e.g., 'Qwen/Qwen2.5-7B-Instruct' should be correct).")
            print("- Run 'pip install --upgrade transformers' to ensure you have the latest version.")
            print(f"- Clear the cache in {cache_dir or '~/.cache/huggingface/hub'} and try again.")
            raise

        print(f"Model configuration: {configuration}")
        logging.info(f"Model configuration: {configuration}")

        from transformers import modeling_utils
        if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

        # Patch configuration to avoid NoneType error
        if not hasattr(configuration, 'parallel_style') or configuration.parallel_style is None:
            configuration.parallel_style = "none"
            logging.warning("Patched configuration.parallel_style to 'none'.")
        if not hasattr(configuration, '_fsdp_config') or configuration._fsdp_config is None:
            configuration._fsdp_config = {}
            logging.warning("Patched configuration._fsdp_config to empty dict.")
        if not hasattr(configuration, 'model_parallel') or configuration.model_parallel is None:
            configuration.model_parallel = False
            logging.warning("Patched configuration.model_parallel to False.")

        # Detect available GPUs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")
        logging.info(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=configuration,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None
        )

        # Wrap model with DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = DataParallel(model)
        model.to(device)
        model.eval()

        print(f"Loading DataFrame from {input_filename}...")
        logging.info(f"Loading DataFrame from {input_filename}...")
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries.")
        logging.info(f"Loaded DataFrame with {len(df)} entries.")
        print(f"DataFrame columns: {df.columns}")
        logging.info(f"DataFrame columns: {df.columns}")

        if full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{full_question_column}' column.")

        print(f"Validating '{full_question_column}' column...")
        logging.info(f"Validating '{full_question_column}' column...")
        invalid_questions = df[full_question_column].apply(lambda x: not isinstance(x, str) or not x.strip())
        if invalid_questions.any():
            invalid_indices = invalid_questions[invalid_questions].index.tolist()
            invalid_samples = df.loc[invalid_indices, full_question_column].head().to_dict()
            raise ValueError(f"Found {len(invalid_indices)} invalid questions in '{full_question_column}' column: {invalid_samples}")

        # Initialize DataFrame columns
        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if sampling > 1:
            df["count_A"] = 0
            df["count_B"] = 0
            df["count_C"] = 0
            df["count_D"] = 0

        print("Testing with first 5 questions...")
        logging.info("Testing with first 5 questions...")
        test_questions = df[full_question_column][:5].tolist()
        test_indices = df.index[:5].tolist()
        answers = process_batch(test_questions, tokenizer, model, sampling, test_indices, device)
        for idx, ans in zip(test_indices, answers):
            print(f"Test question at index {idx}: Answers = {ans}")
            logging.info(f"Test question at index {idx}: Answers = {ans}")
        torch.cuda.empty_cache()

        print("Processing all questions...")
        logging.info("Processing all questions...")
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Initial processing"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_questions = df[full_question_column][start_idx:end_idx].tolist()
            batch_indices = df.index[start_idx:end_idx].tolist()

            # Skip if all answers are valid (for sampling == 1)
            if sampling == 1 and all(df.at[idx, "model_answer"] and is_valid_answer(df.at[idx, "model_answer"]) for idx in batch_indices):
                continue

            batch_answers = process_batch(batch_questions, tokenizer, model, sampling, batch_indices, device)
            for idx, answers in zip(batch_indices, batch_answers):
                if sampling == 1:
                    df.at[idx, "model_answer"] = answers[0]
                else:
                    df.at[idx, "model_answer"] = answers
                    df.at[idx, "count_A"] = answers.count("A")
                    df.at[idx, "count_B"] = answers.count("B")
                    df.at[idx, "count_C"] = answers.count("C")
                    df.at[idx, "count_D"] = answers.count("D")
            torch.cuda.empty_cache()

        retry_count = 0
        while retry_count < max_retries:
            if sampling == 1:
                invalid_indices = df.index[
                    df["model_answer"].isna() |
                    (df["model_answer"] == "") |
                    (df["model_answer"] == "Error") |
                    (~df["model_answer"].apply(is_valid_answer))
                ].tolist()
            else:
                invalid_indices = df.index[
                    df["model_answer"].apply(lambda x: any(a == "" or a == "Error" or not is_valid_answer(a) for a in x))
                ].tolist()

            if not invalid_indices:
                print("All entries have valid answers!")
                logging.info("All entries have valid answers!")
                break

            print(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            logging.info(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            for start_idx in tqdm(range(0, len(invalid_indices), batch_size), desc=f"Retry {retry_count + 1}"):
                batch_indices = invalid_indices[start_idx:start_idx + batch_size]
                batch_questions = df.loc[batch_indices, full_question_column].tolist()
                batch_answers = process_batch(batch_questions, tokenizer, model, sampling, batch_indices, device)
                for idx, answers in zip(batch_indices, batch_answers):
                    if sampling == 1:
                        df.at[idx, "model_answer"] = answers[0]
                    else:
                        df.at[idx, "model_answer"] = answers
                        df.at[idx, "count_A"] = answers.count("A")
                        df.at[idx, "count_B"] = answers.count("B")
                        df.at[idx, "count_C"] = answers.count("C")
                        df.at[idx, "count_D"] = answers.count("D")
                torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        # Construct output directory: output_sampling/{dataset}/{question_type}/{prefix_type}/{prefix_subtype}/{academic_level}
        output_dir_parts = [f"output_sampling/{dataset}"]
        if question_type:
            output_dir_parts.append(question_type)
        if prefix_type:
            output_dir_parts.append(prefix_type)
            output_dir_parts.append(prefix_subtype)
            if prefix_type == "academic":
                output_dir_parts.append(academic_level)
        output_dir = os.path.join(*[part for part in output_dir_parts if part])

        os.makedirs(output_dir, exist_ok=True)

        model_short_name = model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sampling_str = f"_sampling_{sampling}" if sampling > 1 else ""
        output_filename = f"{output_dir}/{model_short_name}{sampling_str}_{timestamp_str}.pkl"

        if sampling == 1:
            invalid_count = len(df[
                df["model_answer"].isna() |
                (df["model_answer"] == "") |
                (df["model_answer"] == "Error") |
                (~df["model_answer"].apply(is_valid_answer))
            ])
        else:
            invalid_count = len(df[
                df["model_answer"].apply(lambda x: any(a == "" or a == "Error" or not is_valid_answer(a) for a in x))
            ])

        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries still have invalid answers after {max_retries} retries.")
            logging.warning(f"{invalid_count} entries still have invalid answers after {max_retries} retries.")
        else:
            print("All entries successfully populated with valid answers!")
            logging.info("All entries successfully populated with valid answers!")

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