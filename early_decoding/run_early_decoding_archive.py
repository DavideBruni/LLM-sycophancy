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
import numpy as np

# Set up logging
logging.basicConfig(filename='inference_early_decoding.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with chain-of-thought and/or logit computation using Transformers.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name for inference (e.g., 'meta-llama/Llama-2-7b-hf', 'Qwen/Qwen2.5-7B-Instruct')")
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
                             "'opinion_only' or 'plain' (no prefix or opinion).")
    parser.add_argument("--input_filename", type=str, default="output/mmlu/mmlu_plain.pkl",
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--full_question_column", type=str, default="full_question",
                        help="Name of the column containing the full question text in the input DataFrame")
    parser.add_argument("--inference_mode", type=str, default="logit_only",
                        choices=["logit_only", "logit_and_cot"],
                        help="Inference mode: 'logit_only' for logit-based answer selection without CoT, "
                             "'logit_and_cot' for chain-of-thought generation and logit-based selection.")
    parser.add_argument("--inference_layer", type=str, default="all",
                        choices=["", "all", "odd", "even", "last"],
                        help="Layers to compute logits: '' or 'all' for all layers, 'odd' for odd-numbered layers, 'even' for "
                             "even-numbered layers, 'last' for only the last layer.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

def get_layer_indices(total_layers, inference_layer_arg):
    """Determine which layer indices to process based on inference_layer argument."""
    if inference_layer_arg in ["", "all"]:
        return list(range(total_layers))
    elif inference_layer_arg == "odd":
        return [i for i in range(total_layers) if (i + 1) % 2 == 1 or i == total_layers - 1]
    elif inference_layer_arg == "even":
        return [i for i in range(total_layers) if (i + 1) % 2 == 0 or i == total_layers - 1]
    elif inference_layer_arg == "last":
        return [total_layers - 1]
    else:
        raise ValueError(f"Invalid inference_layer: {inference_layer_arg}")

def process_question(args, question, tokenizer, model, inference_mode, inference_layer_arg, question_index):
    try:
        logging.debug(f"Processing question at index {question_index}: '{question[:50]}...'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at index {question_index}: must be a non-empty string, got '{question}'")

        # Prompt suffix for model answers
        prompt = f"{question}\nAnswer:"

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        raw_output = ""  # Store generated output if CoT is enabled
        if inference_mode == "logit_and_cot":
            logging.info(f"Generating CoT for index {question_index}: {prompt[:100]}...")
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            raw_output = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            logging.debug(f"Raw model output (CoT): {raw_output}")

        # Compute logits for answer selection at specified layers
        logging.info(f"Computing layer-wise logits for index {question_index}: {prompt[:100]}...")
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (embedding_output, layer_0_output, ..., layer_N-1_output)

        answer_tokens_ids = {} # Correctly initialized here
        for letter in 'ABCD':
            token_ids = []
            # Try plain letter
            plain_token = tokenizer.encode(letter, add_special_tokens=False)
            if plain_token:
                token_ids.append(plain_token[0])
            # Try space-prefixed letter
            space_token = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if space_token and len(space_token) > 0:
                token_ids.append(space_token[0])
            # Remove duplicates and validate
            # FIX: Change 'answer_token_ids' to 'answer_tokens_ids'
            answer_tokens_ids[letter] = sorted(list(set(token_ids)))
            if not answer_tokens_ids[letter]: # Also fix here
                raise ValueError(f"Could not find valid token ID for '{letter}' or ' {letter}'. "
                                 f"Check tokenizer for '{model.name_or_path}'.")
        logging.debug(f"Resolved token IDs: {answer_tokens_ids}") # And here


        # Initialize layer-wise logits storage
        total_layers = model.config.num_hidden_layers
        layer_indices_to_process = get_layer_indices(total_layers, inference_layer_arg)
        layer_wise_abcd_logits = {f"Layer_{i}": {} for i in layer_indices_to_process}

        # Process logits for each specified layer
        for layer_idx in layer_indices_to_process:
            hidden_state_at_last_input_token = hidden_states[layer_idx + 1][:, -1, :]
            layer_logits_raw = model.lm_head(hidden_state_at_last_input_token)
            # FIX: Change 'layer_wise_abcd_logits[f"layer_{layer_idx}"]' to 'layer_wise_abcd_logits[f"Layer_{layer_idx}"]'
            # Also ensure consistency in the variable name used for iteration.
            current_layer_logits = layer_wise_abcd_logits[f"Layer_{layer_idx}"] # Use f"Layer_{layer_idx}" to match dict init

            for letter in 'ABCD':
                max_logit_for_letter = -float('inf')
                for token_id in answer_tokens_ids[letter]: # FIX: Use answer_tokens_ids
                    if token_id < layer_logits_raw.shape[-1]:
                        max_logit_for_letter = max(max_logit_for_letter, layer_logits_raw[0, token_id].item())
                current_layer_logits[letter] = max_logit_for_letter
            logging.debug(f"Layer {layer_idx} ABCD logits: {current_layer_logits}")

        # Determine final answer based on the last layer's logits
        final_layer_idx = total_layers - 1
        if final_layer_idx not in layer_indices_to_process:
            hidden_state_at_last_input_token = hidden_states[final_layer_idx + 1][:, -1, :]
            final_layer_logits_raw = model.lm_head(hidden_state_at_last_input_token)
            last_layer_abcd_logits = {}
            for letter in 'ABCD':
                max_logit_for_letter = -float('inf')
                for token_id in answer_tokens_ids[letter]: # FIX: Use answer_tokens_ids
                    if token_id < final_layer_logits_raw.shape[-1]:
                        max_logit_for_letter = max(max_logit_for_letter, final_layer_logits_raw[0, token_id].item())
                last_layer_abcd_logits[letter] = max_logit_for_letter
        else:
            # FIX: Use f"Layer_{final_layer_idx}" to match dict init
            last_layer_abcd_logits = layer_wise_abcd_logits[f"Layer_{final_layer_idx}"]


        # Convert logits to probabilities for answer selection
        abcd_logit_values = torch.tensor([last_layer_abcd_logits[letter] for letter in 'ABCD'])
        abcd_probs = torch.softmax(abcd_logit_values, dim=-1).tolist()
        answer_probs_dict = {letter: prob for letter, prob in zip('ABCD', abcd_probs)}
        selected_answer = max(answer_probs_dict, key=answer_probs_dict.get)
        logging.debug(f"Final answer based on last layer logits: {selected_answer}, Probabilities: {answer_probs_dict}")

        if not is_valid_answer(selected_answer):
            logging.warning(f"Invalid final selected answer at index {question_index}: '{selected_answer}' for question: '{question[:50]}...'")
            return "Error", layer_wise_abcd_logits, raw_output

        return selected_answer, layer_wise_abcd_logits, raw_output

    except Exception as e:
        logging.error(f"Error processing question at index {question_index} '{question[:50]}...': {str(e)}\n{traceback.format_exc()}")
        return "Error", {}, "Error in processing"

def main():
    try:
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

        # Check HF_TOKEN
        hf_token = config.HF_TOKEN
        if not hf_token or hf_token == "YOUR_HF_TOKEN_HERE":
            raise ValueError("HF_TOKEN is not set or is still the placeholder. Please set it in config.py or as an environment variable.")
        logging.info(f"HF_TOKEN is set (length: {len(hf_token)} characters)")

        # Debug: Log parsed arguments
        print(f"Parsed model_name: '{model_name}'")
        logging.info(f"Parsed model_name: '{model_name}'")
        if not model_name or not isinstance(model_name, str) or model_name.strip() == "":
            raise ValueError(f"Model name is invalid or empty: '{model_name}'. Please provide a valid model name (e.g., 'Qwen/Qwen2.5-7B-Instruct').")

        # Debug: Environment and library info
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA version: {torch.version.cuda}")

        # Load tokenizer
        print("Loading tokenizer...")
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info("Tokenizer loaded successfully")

                
        # Load and inspect model configuration
        print("Loading configuration...")
        logging.info("Loading configuration...")
        model_config = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

        # --- Enhanced Patching and Debugging ---
        # Ensure these are not None. The 'or' condition ensures it's set if currently None.
        model_config.tensor_parallel_degree = getattr(model_config, 'tensor_parallel_degree', None) or 1
        logging.info(f"Patched tensor_parallel_degree to {model_config.tensor_parallel_degree}")

        model_config.pipeline_parallel_degree = getattr(model_config, 'pipeline_parallel_degree', None) or 1
        logging.info(f"Patched pipeline_parallel_degree to {model_config.pipeline_parallel_degree}")

        # This is often the culprit. Ensure it's not None.
        # 'eager' is a safe default for non-optimized attention.
        model_config._attn_implementation = getattr(model_config, '_attn_implementation', None) or 'eager'
        logging.info(f"Patched _attn_implementation to '{model_config._attn_implementation}'")

        # Check for other potential None values that might be iterated
        # You can inspect the Qwen2Config source or common Transformers patterns
        # Example: If there was a 'parallel_strategy' attribute that could be None
        # model_config.parallel_strategy = getattr(model_config, 'parallel_strategy', None) or 'default'
        # logging.info(f"Patched parallel_strategy to '{model_config.parallel_strategy}'")

        print(f"Model config after patching: {model_config}") # Print the patched config
        logging.info(f"Model config after patching: {model_config}")

        # Load model
        print("Loading Transformers model...")
        logging.info("Loading Transformers model...")
        device = "cpu" # Force CPU for debugging
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config, # Pass the *patched* config
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16 # Use float16 for CPU compatibility
        )

        model.to(device)
        model.eval()
        if tokenizer.pad_token == '[PAD]':
            model.resize_token_embeddings(len(tokenizer))
        logging.info(f"Model loaded successfully on {device}")

        # Load DataFrame
        print(f"Loading DataFrame from {input_filename}...")
        logging.info(f"Loading DataFrame from {input_filename}...")
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries.")
        logging.info(f"Loaded DataFrame with {len(df)} entries.")
        print(f"DataFrame columns: {df.columns}")
        logging.info(f"DataFrame columns: {df.columns}")

        if full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{full_question_column}' column.")

        # Validate questions
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

        # Test with first 5 questions
        print("Testing with first 5 questions...")
        logging.info("Testing with first 5 questions...")
        for idx in df.index[:5]:
            question = df.at[idx, full_question_column]
            answer, layer_logits_data, raw_out = process_question(
                args, # <--- ADD THIS ARGUMENT
                question, tokenizer, model, inference_mode, inference_layer, idx
            )
            print(f"Test question at index {idx}: Answer = {answer}, Layer Logits Data = {layer_logits_data}, Raw Output = {raw_out[:100]}...")
            logging.info(f"Test question at index {idx}: Answer = {answer}, Layer Logits Data = {layer_logits_data}, Raw Output = {raw_out}")

        # Process all questions
        print("Processing all questions...")
        logging.info("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]) or df.at[idx, "layer_logits"] is None:
                question = df.at[idx, full_question_column]
                answer, layer_logits_data, raw_out = process_question(
                    args, # <--- ADD THIS ARGUMENT HERE TOO
                    question, tokenizer, model, inference_mode, inference_layer, idx
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits_data
                df.at[idx, "raw_output"] = raw_out


        # Retry logic for failed entries
        retry_count = 0
        while retry_count < max_retries:
            invalid_indices = df.index[
                df["model_answer"].isna() |
                (df["model_answer"] == "") |
                (df["model_answer"] == "Error") |
                (~df["model_answer"].apply(is_valid_answer)) |
                (df["layer_logits"].apply(lambda x: not x))
            ].tolist()

            if not invalid_indices:
                print("All entries have valid answers and layer logits collected!")
                logging.info("All entries have valid answers and layer logits collected!")
                break

            print(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers or missing logits.")
            logging.info(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers or missing logits.")
            # Make sure to add `args` in the process_question call inside the retry loop as well!
            for idx in tqdm(invalid_indices, desc=f"Retry {retry_count + 1}"):
                question = df.at[idx, full_question_column]
                answer, layer_logits_data, raw_out = process_question(
                    args, # <--- AND HERE
                    question, tokenizer, model, inference_mode, inference_layer, idx
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits_data
                df.at[idx, "raw_output"] = raw_out

            retry_count += 1
            time.sleep(1)

        # Construct output directory and filename
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

        model_short_name = model_name.split("/")[-1].replace(".", "_").replace("-", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_mode_str = 'cot' if inference_mode == 'logit_and_cot' else 'logit'
        output_filename = f"{output_dir}/{model_short_name}_{inference_mode_str}_{inference_layer}_{timestamp_str}.pkl"

        invalid_count = len(df[
            df["model_answer"].isna() |
            (df["model_answer"] == "") |
            (df["model_answer"] == "Error") |
            (~df["model_answer"].apply(is_valid_answer)) |
            (df["layer_logits"].apply(lambda x: not x))
        ])
        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries still have invalid answers or missing layer logits after {max_retries} retries.")
            logging.warning(f"{invalid_count} entries still have invalid answers or missing layer logits after {max_retries} retries.")
        else:
            print("All entries successfully populated with valid answers and layer logits!")
            logging.info("All entries successfully populated with valid answers and layer logits!")

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