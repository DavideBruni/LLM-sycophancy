# This script runs inference on a dataset using the vLLM library with LLaMA & Qwen models.

import torch
import pandas as pd
import config
from vllm import LLM, SamplingParams
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import logging
import os
import traceback
from transformers import AutoTokenizer  # Used only for token ID mapping
import torch.distributed as dist

# Set up logging for the script
logging.basicConfig(filename='inference.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with chain-of-thought and/or logit computation using vLLM.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name for inference (must be vLLM-compatible)")
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
    parser.add_argument("--input_filename", type=str, default="output/mmlu/mmlu_plain.pkl", 
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--full_question_column", type=str, default="full_question", 
                        help="Name of the column containing the full question text in the input DataFrame")
    parser.add_argument("--inference_mode", type=str, default="logit_and_cot", 
                        choices=["logit_only", "logit_and_cot"], 
                        help="Inference mode: 'logit_only' for logit-based answer selection only, "
                             "'logit_and_cot' for chain-of-thought generation and logit-based selection.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def process_question(question, tokenizer, llm, sampling_params_cot, sampling_params_logit, inference_mode, question_index):
    """Process a single question, return the raw output, answer based on logits, raw logits for A, B, C, D, and top 10 logits."""
    try:
        logging.debug(f"Processing question at index {question_index}: '{question[:50]}...'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at index {question_index}: must be a non-empty string, got '{question}'")

        # Set prompt based on inference mode
        if inference_mode == "logit_and_cot":
            prompt = (f"Question: ||{question}||\n"
                      "You MUST explain your detailed reasoning step by step to determine the correct answer. "
                      "After explaining your reasoning, provide your final answer in the format {A}, {B}, {C}, or {D}. "
                      "The answer must be exactly one uppercase letter (A, B, C, D) enclosed in curly braces. "
                      "Do not include any additional text after the curly braces.")
        else:  # logit_only
            prompt = f"Question: ||{question}||\nRespond with exactly one uppercase letter (A, B, C, D) and nothing else.\nAnswer:"

        # Generate CoT output for logit_and_cot mode
        raw_output = ""
        if inference_mode == "logit_and_cot":
            logging.info(f"Generating CoT for index {question_index}: {prompt[:100]}...")
            outputs = llm.generate([prompt], sampling_params_cot,use_tqdm=False)
            logging.info(f"vLLM CoT output for index {question_index}: {outputs}")
            if outputs and outputs[0].outputs:
                decoded_output = outputs[0].outputs[0].text
                # Extract raw output after the double-pipe question block
                if "||" in prompt:
                    parts = (prompt + decoded_output).split("||")
                    if len(parts) >= 3:
                        raw_output = parts[2].lstrip()
                    else:
                        raw_output = decoded_output
                else:
                    raw_output = decoded_output
            else:
                raw_output = "No output generated"
            logging.debug(f"Raw model output: {raw_output}")

        # Compute logits for answer selection
        sampling_params_logit.logprobs = 100  # Request top 100 logprobs for logit computation
        logging.info(f"Generating logits for index {question_index}: {prompt[:100]}...")
        outputs = llm.generate([prompt], sampling_params_logit, use_tqdm=False)
        logging.info(f"vLLM logit output for index {question_index}: {outputs}")
        if not outputs or not outputs[0].outputs:
            logging.warning(f"No output generated for logits at index {question_index}")
            return "Error", {}, "Error in processing", {}

        # Extract logprobs for the last token
        logprobs = outputs[0].outputs[0].logprobs
        if not logprobs:
            logging.warning(f"No logprobs returned for index {question_index}")
            return "Error", {}, raw_output, {}

        last_token_logprobs = logprobs[-1]  # Logprobs for the last generated token
        logging.debug(f"Last token logprobs: {[(token, prob.logprob, prob.decoded_token) for token, prob in last_token_logprobs.items()][:20]}")

        # Map tokens to logprobs
        answer_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCD'}
        space_answer_tokens = {letter: tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in 'ABCD'}
        logging.debug(f"Answer token IDs: {answer_tokens}")
        logging.debug(f"Space-prefixed answer token IDs: {space_answer_tokens}")

        # Convert logprobs to raw logits, handling both letter and space-prefixed letter
        answer_logits = {'A': float('-inf'), 'B': float('-inf'), 'C': float('-inf'), 'D': float('-inf')}
        for token, logprob in last_token_logprobs.items():
            try:
                token_ids = tokenizer.encode(logprob.decoded_token, add_special_tokens=False)
                if not token_ids:
                    continue  # Skip empty encodings
                token_id = token_ids[0]
            except Exception as e:
                logging.warning(f"Skipping token '{logprob.decoded_token}' due to encoding error: {e}")
                continue

            for letter in 'ABCD':
                if token_id == answer_tokens[letter] or token_id == space_answer_tokens[letter]:
                    answer_logits[letter] = max(answer_logits[letter], logprob.logprob)


        logging.debug(f"Raw logits for A, B, C, D: {answer_logits}")

        # Extract top 10 logits (excluding A, B, C, D and their space-prefixed versions)
        exclude_token_ids = set(answer_tokens.values()) | set(space_answer_tokens.values())
        top_10_logits = {}
        sorted_logprobs = sorted(last_token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
        for token_id, logprob in sorted_logprobs:
            try:
                token_ids = tokenizer.encode(logprob.decoded_token, add_special_tokens=False)
                if not token_ids:
                    continue
                if token_ids[0] in exclude_token_ids:
                    continue
                decoded = tokenizer.convert_ids_to_tokens(token_ids[0])
                top_10_logits[decoded] = logprob.logprob
                if len(top_10_logits) >= 10:
                    break
            except Exception as e:
                logging.warning(f"Skipping token '{logprob.decoded_token}' due to decoding error: {e}")
                continue

        logging.debug(f"Top 10 logits (excluding A, B, C, D and space-prefixed versions): {top_10_logits}")

        # Compute softmax probabilities for answer selection
        answer_probs = {}
        for letter, logprob in answer_logits.items():
            if logprob != float('-inf'):
                answer_probs[letter] = torch.exp(torch.tensor(logprob)).item()
            else:
                answer_probs[letter] = 0.0

        total_prob = sum(answer_probs.values())
        if total_prob > 0:
            answer_probs = {letter: prob / total_prob for letter, prob in answer_probs.items()}
        else:
            answer_probs = {letter: 0.25 for letter in 'ABCD'}

        selected_answer = max(answer_probs, key=answer_probs.get)
        logging.debug(f"Answer based on probabilities: {selected_answer}, Probabilities: {answer_probs}")

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
    dataset = args.dataset
    prefix_type = args.prefix_type
    academic_level = args.academic_level
    prefix_subtype = args.prefix_subtype
    question_type = args.question_type
    input_filename = args.input_filename
    full_question_column = args.full_question_column
    inference_mode = args.inference_mode
    max_retries = args.max_retries

    # Validation: Ensure academic_level is only specified for academic prefix
    if academic_level and prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")

    if question_type == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")

    hf_token = config.HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")

    try:
        print("Loading tokenizer for token mapping...")
        logging.info("Loading tokenizer for token mapping...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

        print("Loading vLLM model...")
        logging.info("Loading vLLM model...")
        os.environ['HF_TOKEN'] = hf_token
        llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.8, dtype="float32", max_logprobs=100, max_model_len=4096)

        # Define sampling parameters
        sampling_params_cot = SamplingParams(
            max_tokens=1000,
            temperature=0.0,
            top_p=1.0,
            include_stop_str_in_output=True
        )
        sampling_params_logit = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=100,
            include_stop_str_in_output=True
        )

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

        if full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame '{input_filename}' must contain a '{full_question_column}' column.")

        print(f"Validating '{full_question_column}' column contents...")
        logging.info(f"Validating '{full_question_column}' column contents...")
        invalid_questions = df[full_question_column].apply(lambda x: not isinstance(x, str) or not x.strip())
        if invalid_questions.any():
            invalid_indices = invalid_questions[invalid_questions].index.tolist()
            invalid_samples = df.loc[invalid_indices, full_question_column].head().to_dict()
            raise ValueError(f"Found {len(invalid_indices)} invalid questions in '{full_question_column}' column (must be non-empty strings). First few: {invalid_samples}")

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
            question = df.at[idx, full_question_column]
            answer, logits, raw_out, top_10 = process_question(
                question, tokenizer, llm, sampling_params_cot, sampling_params_logit, inference_mode, idx
            )
            print(f"Test question at index {idx}: Answer = {answer}, Logits = {logits}, Raw Output = {raw_out[:100]}..., Top 10 Logits = {top_10}")
            logging.info(f"Test question at index {idx}: Answer = {answer}, Logits = {logits}, Raw Output = {raw_out}, Top 10 Logits = {top_10}")
            torch.cuda.empty_cache()

        print("Processing all questions...")
        logging.info("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]):
                question = df.at[idx, full_question_column]
                answer, logits, raw_out, top_10 = process_question(
                    question, tokenizer, llm, sampling_params_cot, sampling_params_logit, inference_mode, idx
                )
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
                question = df.at[idx, full_question_column]
                answer, logits, raw_out, top_10 = process_question(
                    question, tokenizer, llm, sampling_params_cot, sampling_params_logit, inference_mode, idx
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "answer_logits"] = logits
                df.at[idx, "raw_output"] = raw_out
                df.at[idx, "top_10_logits"] = top_10
                torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        # Construct output directory dynamically: output_inference/{dataset}/{question_type}/{prefix_type}/{prefix_subtype}/{academic_level}
        output_dir_parts = [f"output_inference/{dataset}"]
        if question_type:
            output_dir_parts.append(question_type)
        if prefix_type:
            output_dir_parts.append(prefix_type)
            output_dir_parts.append(prefix_subtype)
            if prefix_type == "academic":
                output_dir_parts.append(academic_level)
        output_dir = os.path.join(*output_dir_parts)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define output filename based on inference mode
        model_short_name = model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if inference_mode == "logit_and_cot":
            output_filename = f"{output_dir}/{model_short_name}_cot_{timestamp_str}.pkl"
        else:  # logit_only
            output_filename = f"{output_dir}/{model_short_name}_logit_{timestamp_str}.pkl"

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
    if dist.is_initialized():
        dist.destroy_process_group()