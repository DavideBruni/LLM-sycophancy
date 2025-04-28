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
from transformers import AutoTokenizer, AutoModelForCausalLM
import uuid
import numpy as np

# Set up logging
logging.basicConfig(filename='inference.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with chain-of-thought and/or logit computation using Transformers.")
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
    parser.add_argument("--input_filename", type=str, default="output/mmlu/mmlu_plain.pkl", 
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--full_question_column", type=str, default="full_question", 
                        help="Name of the column containing the full question text in the input DataFrame")
    parser.add_argument("--inference_mode", type=str, default="logit_and_cot", 
                        choices=["logit_only", "logit_and_cot"], 
                        help="Inference mode: 'logit_only' for logit-based answer selection without CoT, "
                             "'logit_and_cot' for chain-of-thought generation and logit-based selection.")
    parser.add_argument("--inference_layer", type=str, default="last", 
                        choices=["", "all", "odd", "even", "last"], 
                        help="Layers to compute logits: '' or 'all' for all layers, 'odd' for odd-numbered layers (including last), "
                             "'even' for even-numbered layers (including last), 'last' for last layer only.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def get_layer_indices(total_layers, inference_layer):
    """Determine which layer indices to process based on inference_layer argument."""
    if inference_layer in ["", "all"]:
        return list(range(total_layers))
    elif inference_layer == "odd":
        return [i for i in range(total_layers) if i % 2 == 1 or i == total_layers - 1]
    elif inference_layer == "even":
        return [i for i in range(total_layers) if i % 2 == 0 or i == total_layers - 1]
    elif inference_layer == "last":
        return [total_layers - 1]
    else:
        raise ValueError(f"Invalid inference_layer: {inference_layer}")

def process_question(question, tokenizer, model, inference_mode, inference_layer, question_index):
    try:
        logging.debug(f"Processing question at index {question_index}: '{question[:50]}...'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at index {question_index}: must be a non-empty string, got '{question}'")

        # Set prompt based on mode
        if inference_mode == "logit_and_cot":
            prompt = (f"Question: ||{question}||\n"
                      "You MUST explain your detailed reasoning step by step to determine the correct answer. "
                      "After explaining your reasoning, provide your final answer in the format {A}, {B}, {C}, or {D}. "
                      "The answer must be exactly one uppercase letter (A, B, C, D) enclosed in curly braces. "
                      "Do not include any additional text after the curly braces.")
        else:  # logit_only
            prompt = f"Question: ||{question}||\nRespond with exactly one uppercase letter (A, B, C, D) and nothing else.\nAnswer:"

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate CoT output for logit_and_cot mode
        raw_output = ""
        if inference_mode == "logit_and_cot":
            logging.info(f"Generating CoT for index {question_index}: {prompt[:100]}...")
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            decoded_output = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
            if "||" in prompt:
                parts = (prompt + decoded_output).split("||")
                raw_output = parts[2].lstrip() if len(parts) >= 3 else decoded_output
            else:
                raw_output = decoded_output
            logging.debug(f"Raw model output: {raw_output}")

        # Compute logits for answer selection
        logging.info(f"Computing logits for index {question_index}: {prompt[:100]}...")
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of hidden states for each layer
            logits = outputs.logits[:, -1, :]  # Logits for the last token

        # Map answer tokens
        answer_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCD'}
        space_answer_tokens = {letter: tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in 'ABCD'}
        logging.debug(f"Answer token IDs: {answer_tokens}")
        logging.debug(f"Space-prefixed answer token IDs: {space_answer_tokens}")

        # Initialize layer-wise logits storage
        total_layers = len(hidden_states) - 1  # Number of transformer layers
        layer_indices = get_layer_indices(total_layers, inference_layer)
        layer_logits = {f"layer_{i}": {'A': float('-inf'), 'B': float('-inf'), 'C': float('-inf'), 'D': float('-inf')} 
                        for i in layer_indices}

        # Process logits for each specified layer
        for layer_idx in layer_indices:
            # Get hidden states for the layer (last token)
            hidden_state = hidden_states[layer_idx + 1][:, -1, :]  # +1 because hidden_states includes input embeddings
            # Project hidden state to logits using the model's language model head
            layer_logits_raw = model.lm_head(hidden_state)
            answer_logits = layer_logits[f"layer_{layer_idx}"]

            # Extract logits for A, B, C, D
            for letter in 'ABCD':
                token_id = answer_tokens[letter]
                space_token_id = space_answer_tokens[letter]
                logit = max(
                    layer_logits_raw[0, token_id].item(),
                    layer_logits_raw[0, space_token_id].item()
                )
                answer_logits[letter] = logit

            logging.debug(f"Layer {layer_idx} logits for A, B, C, D: {answer_logits}")

        # Use last layer logits for answer selection
        last_layer_logits = layer_logits[f"layer_{total_layers-1}"]
        answer_probs = {}
        for letter, logprob in last_layer_logits.items():
            answer_probs[letter] = torch.exp(torch.tensor(logprob)).item() if logprob != float('-inf') else 0.0

        total_prob = sum(answer_probs.values())
        if total_prob > 0:
            answer_probs = {letter: prob / total_prob for letter, prob in answer_probs.items()}
        else:
            answer_probs = {letter: 0.25 for letter in 'ABCD'}

        selected_answer = max(answer_probs, key=answer_probs.get)
        logging.debug(f"Answer based on probabilities: {selected_answer}, Probabilities: {answer_probs}")

        if not is_valid_answer(selected_answer):
            logging.warning(f"Invalid answer at index {question_index}: '{selected_answer}' for question: '{question[:50]}...'")
            return "Error", layer_logits, raw_output

        return selected_answer, layer_logits, raw_output

    except Exception as e:
        logging.error(f"Error processing question at index {question_index} '{question[:50]}...': {str(e)}\n{traceback.format_exc()}")
        return "Error", {}, "Error in processing"

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
    inference_layer = args.inference_layer
    max_retries = args.max_retries

    # Validation
    if academic_level and prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")
    if question_type == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")

    hf_token = config.HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")

    try:
        print("Loading tokenizer...")
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

        print("Loading Transformers model...")
        logging.info("Loading Transformers model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float32,
            output_hidden_states=True
        )
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
        if "layer_logits" not in df.columns:
            df["layer_logits"] = None
        if "raw_output" not in df.columns:
            df["raw_output"] = None

        print("Testing with first 5 questions...")
        logging.info("Testing with first 5 questions...")
        for idx in df.index[:5]:
            question = df.at[idx, full_question_column]
            answer, layer_logits, raw_out = process_question(
                question, tokenizer, model, inference_mode, inference_layer, idx
            )
            print(f"Test question at index {idx}: Answer = {answer}, Layer Logits = {layer_logits}, Raw Output = {raw_out[:100]}...")
            logging.info(f"Test question at index {idx}: Answer = {answer}, Layer Logits = {layer_logits}, Raw Output = {raw_out}")
            torch.cuda.empty_cache()

        print("Processing all questions...")
        logging.info("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]):
                question = df.at[idx, full_question_column]
                answer, layer_logits, raw_out = process_question(
                    question, tokenizer, model, inference_mode, inference_layer, idx
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_out
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
                answer, layer_logits, raw_out = process_question(
                    question, tokenizer, model, inference_mode, inference_layer, idx
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_out
                torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        # Construct output directory: output_inference/{dataset}/{question_type}/{prefix_type}/{prefix_subtype}/{academic_level}
        output_dir_parts = [f"output_inference/{dataset}"]
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
        inference_mode_str = 'cot' if inference_mode == 'logit_and_cot' else 'logit'
        # Include inference_layer in the filename
        output_filename = f"{output_dir}/{model_short_name}_{inference_mode_str}_{inference_layer}_{timestamp_str}.pkl"

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