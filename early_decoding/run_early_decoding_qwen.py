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

# Minimal logging setup - only errors and warnings
logging.basicConfig(filename='inference_early_decoding.log', level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with chain-of-thought and/or logit computation using Transformers.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name for inference")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu"], help="Dataset to use")
    parser.add_argument("--prefix_type", type=str, default="", choices=["academic", "behavior", ""], help="Type of prefix used")
    parser.add_argument("--academic_level", type=str, default="", choices=["beginner", "intermediate", "advanced", ""], help="Academic level for academic prefix")
    parser.add_argument("--prefix_subtype", type=str, default="", choices=["original", "mixing_subject", "third_pov", ""], help="Subtype of prefix")
    parser.add_argument("--question_type", type=str, default="plain", choices=["prefix_and_opinion", "opinion_only", "plain"], help="Type of the questions")
    parser.add_argument("--input_filename", type=str, default="output/mmlu/mmlu_plain.pkl", help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--full_question_column", type=str, default="full_question", help="Name of the column containing the full question text")
    parser.add_argument("--inference_mode", type=str, default="logit_only", choices=["logit_only", "logit_and_cot"], help="Inference mode")
    parser.add_argument("--inference_layer", type=str, default="all", choices=["", "all", "odd", "even", "last"], help="Layers to compute logits")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
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

# Pre-compute answer token IDs once globally
def get_answer_token_ids(tokenizer):
    """Pre-compute answer token IDs for efficiency."""
    answer_tokens_ids = {}
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
        # Remove duplicates
        answer_tokens_ids[letter] = sorted(list(set(token_ids)))
        if not answer_tokens_ids[letter]:
            raise ValueError(f"Could not find valid token ID for '{letter}' or ' {letter}'.")
    return answer_tokens_ids

def process_question_batch(args, questions, tokenizer, model, inference_mode, inference_layer_arg, answer_tokens_ids, layer_indices_to_process, total_layers):
    """Process a batch of questions for better efficiency."""
    batch_results = []
    
    try:
        # Prepare prompts
        prompts = [f"{question}\nAnswer:" for question in questions]
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        batch_raw_outputs = [""] * len(questions)
        
        # CoT generation if needed
        if inference_mode == "logit_and_cot":
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            for i, seq in enumerate(outputs):
                batch_raw_outputs[i] = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Compute logits efficiently
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Process each question in the batch
        for batch_idx, question in enumerate(questions):
            try:
                # Initialize layer-wise logits storage
                layer_wise_abcd_logits = {f"Layer_{i}": {} for i in layer_indices_to_process}
                
                # Get sequence length for this specific question (handle padding)
                seq_len = attention_mask[batch_idx].sum().item()
                last_token_idx = seq_len - 1
                
                # Process logits for each specified layer
                for layer_idx in layer_indices_to_process:
                    hidden_state_at_last_token = hidden_states[layer_idx + 1][batch_idx, last_token_idx, :].unsqueeze(0)
                    layer_logits_raw = model.lm_head(hidden_state_at_last_token)
                    current_layer_logits = layer_wise_abcd_logits[f"Layer_{layer_idx}"]
                    
                    for letter in 'ABCD':
                        max_logit_for_letter = -float('inf')
                        for token_id in answer_tokens_ids[letter]:
                            if token_id < layer_logits_raw.shape[-1]:
                                max_logit_for_letter = max(max_logit_for_letter, layer_logits_raw[0, token_id].item())
                        current_layer_logits[letter] = max_logit_for_letter
                
                # Determine final answer based on the last layer's logits
                final_layer_idx = total_layers - 1
                if final_layer_idx not in layer_indices_to_process:
                    hidden_state_at_last_token = hidden_states[final_layer_idx + 1][batch_idx, last_token_idx, :].unsqueeze(0)
                    final_layer_logits_raw = model.lm_head(hidden_state_at_last_token)
                    last_layer_abcd_logits = {}
                    for letter in 'ABCD':
                        max_logit_for_letter = -float('inf')
                        for token_id in answer_tokens_ids[letter]:
                            if token_id < final_layer_logits_raw.shape[-1]:
                                max_logit_for_letter = max(max_logit_for_letter, final_layer_logits_raw[0, token_id].item())
                        last_layer_abcd_logits[letter] = max_logit_for_letter
                else:
                    last_layer_abcd_logits = layer_wise_abcd_logits[f"Layer_{final_layer_idx}"]
                
                # Convert logits to probabilities for answer selection
                abcd_logit_values = torch.tensor([last_layer_abcd_logits[letter] for letter in 'ABCD'])
                abcd_probs = torch.softmax(abcd_logit_values, dim=-1)
                answer_probs_dict = {letter: prob.item() for letter, prob in zip('ABCD', abcd_probs)}
                selected_answer = max(answer_probs_dict, key=answer_probs_dict.get)
                
                if not is_valid_answer(selected_answer):
                    selected_answer = "Error"
                
                batch_results.append((selected_answer, layer_wise_abcd_logits, batch_raw_outputs[batch_idx]))
                
            except Exception as e:
                logging.error(f"Error processing question in batch: {str(e)}")
                batch_results.append(("Error", {}, "Error in processing"))
    
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        # Return error results for all questions in batch
        for _ in questions:
            batch_results.append(("Error", {}, "Error in processing"))
    
    return batch_results

def main():
    try:
        args = parse_args()
        
        # Validation
        if args.academic_level and args.prefix_type != "academic":
            raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")
        if args.question_type == "prefix_and_opinion" and not args.prefix_type:
            raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type must be specified.")
        
        # Check HF_TOKEN
        hf_token = config.HF_TOKEN
        if not hf_token or hf_token == "YOUR_HF_TOKEN_HERE":
            raise ValueError("HF_TOKEN is not set. Please set it in config.py.")
        
        print(f"Loading model: {args.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model configuration with patches
        model_config = AutoConfig.from_pretrained(args.model_name, token=hf_token, trust_remote_code=True)
        model_config.tensor_parallel_degree = getattr(model_config, 'tensor_parallel_degree', None) or 1
        model_config.pipeline_parallel_degree = getattr(model_config, 'pipeline_parallel_degree', None) or 1
        model_config._attn_implementation = getattr(model_config, '_attn_implementation', None) or 'eager'
        
        # Load model - use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=model_config,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model.to(device)
        model.eval()
        
        if tokenizer.pad_token == '[PAD]':
            model.resize_token_embeddings(len(tokenizer))
        
        # Pre-compute answer token IDs
        answer_tokens_ids = get_answer_token_ids(tokenizer)
        
        # Load DataFrame
        print(f"Loading data from {args.input_filename}")
        df = pd.read_pickle(args.input_filename)
        print(f"Loaded {len(df)} questions")
        
        if args.full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{args.full_question_column}' column.")
        
        # Initialize DataFrame columns
        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "layer_logits" not in df.columns:
            df["layer_logits"] = None
        if "raw_output" not in df.columns:
            df["raw_output"] = None
        
        # Get layer configuration
        total_layers = model.config.num_hidden_layers
        layer_indices_to_process = get_layer_indices(total_layers, args.inference_layer)
        
        print("Processing questions...")
        
        # Process in batches for efficiency
        batch_size = args.batch_size
        total_questions = len(df)
        
        for start_idx in tqdm(range(0, total_questions, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, total_questions)
            batch_indices = df.index[start_idx:end_idx]
            
            # Skip if already processed
            batch_to_process = []
            batch_df_indices = []
            
            for idx in batch_indices:
                if not is_valid_answer(df.at[idx, "model_answer"]) or df.at[idx, "layer_logits"] is None:
                    batch_to_process.append(df.at[idx, args.full_question_column])
                    batch_df_indices.append(idx)
            
            if batch_to_process:
                batch_results = process_question_batch(
                    args, batch_to_process, tokenizer, model, args.inference_mode, 
                    args.inference_layer, answer_tokens_ids, layer_indices_to_process, total_layers
                )
                
                # Update DataFrame with results
                for df_idx, (answer, layer_logits, raw_output) in zip(batch_df_indices, batch_results):
                    df.at[df_idx, "model_answer"] = answer
                    df.at[df_idx, "layer_logits"] = layer_logits
                    df.at[df_idx, "raw_output"] = raw_output
        
        # Retry logic for failed entries (simplified)
        for retry in range(args.max_retries):
            invalid_indices = df.index[
                df["model_answer"].isna() |
                (df["model_answer"] == "") |
                (df["model_answer"] == "Error") |
                (~df["model_answer"].apply(is_valid_answer))
            ].tolist()
            
            if not invalid_indices:
                break
            
            print(f"Retry {retry + 1}: {len(invalid_indices)} failed entries")
            
            # Process failed entries in smaller batches
            for start_idx in range(0, len(invalid_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(invalid_indices))
                batch_indices = invalid_indices[start_idx:end_idx]
                batch_questions = [df.at[idx, args.full_question_column] for idx in batch_indices]
                
                batch_results = process_question_batch(
                    args, batch_questions, tokenizer, model, args.inference_mode,
                    args.inference_layer, answer_tokens_ids, layer_indices_to_process, total_layers
                )
                
                for df_idx, (answer, layer_logits, raw_output) in zip(batch_indices, batch_results):
                    df.at[df_idx, "model_answer"] = answer
                    df.at[df_idx, "layer_logits"] = layer_logits
                    df.at[df_idx, "raw_output"] = raw_output
        
        # Save results
        output_dir_parts = [f"output_inference/{args.dataset}"]
        if args.question_type:
            output_dir_parts.append(args.question_type)
        if args.prefix_type:
            output_dir_parts.append(args.prefix_type)
            output_dir_parts.append(args.prefix_subtype)
            if args.prefix_type == "academic":
                output_dir_parts.append(args.academic_level)
        output_dir = os.path.join(*[part for part in output_dir_parts if part])
        os.makedirs(output_dir, exist_ok=True)
        
        model_short_name = args.model_name.split("/")[-1].replace(".", "_").replace("-", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_mode_str = 'cot' if args.inference_mode == 'logit_and_cot' else 'logit'
        output_filename = f"{output_dir}/{model_short_name}_{inference_mode_str}_{args.inference_layer}_{timestamp_str}.pkl"
        
        invalid_count = len(df[df["model_answer"] == "Error"])
        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries failed processing")
        else:
            print("All entries processed successfully!")
        
        df.to_pickle(output_filename)
        print(f"Results saved to {output_filename}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()