import pandas as pd
import requests
import time
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from dataloader import *
from prompt import *
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# Configuration settings
CONFIG = {
    "model_path": "YOUR/LOCAL/LLAMA3/PATH",  # Replace with the actual local path to your Llama-3 model
    "input_file": "patch_db.json",
    "output_file": "llama_Pllm_security_patch_predictions.json",
    "final_metrics_file": "llama_Pllm_final_experiment_metrics.txt",
    "batch_size": 500,  # Batch size parameter (retained for context)
    "retry_limit": 3,
    "request_interval": 1.5,  # Seconds between requests
}

# Global loading of Llama model and tokenizer
print(f"Loading model from: {CONFIG['model_path']}")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_path"],
    # Removed use_auth_token parameter
    trust_remote_code=True
)

# Set padding token to end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_path"],
    device_map="auto",  # Automatically assign device
    torch_dtype=torch.bfloat16,  # Use half-precision to reduce VRAM usage
    low_cpu_mem_usage=True  # Memory optimization mode
)


def predict_with_Llama(prompts, df):
    """Performs prediction using the locally loaded Llama model"""
    predictions = []

    with tqdm(total=len(prompts), desc="Predicting with Llama") as pbar:
        for i, prompt in enumerate(prompts):
            # Construct the formatted prompt
            input_prompt = prompt

            # Encode input
            inputs = tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192  # Limit input length
            )

            # Generation parameters (consistent with temperature and max tokens)
            outputs = model.generate(
                **inputs.to(model.device),  # Ensure input is on the correct device
                max_new_tokens=512,
                temperature=0.10,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,  # Use eos_token as pad_token
                do_sample=True
            )

            # Decode and clean the output
            # Only consider the part after the prompt (the generation)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Since the full prompt is in the output, we extract the generated part (assuming "Answer:" separates them)
            # This extraction logic depends heavily on the 'prompt' structure defined in 'prompt.py'
            try:
                # Assuming the model is instructed to output 'Answer: Yes/No'
                output_text = output_text.split("Answer:")[-1].strip()
            except:
                # Fallback if 'Answer:' separator is not found
                output_text = output_text

            # print(f"Model Output: {output_text}") # Uncomment for debugging

            # Parse results
            pred = parse_answer(output_text)
            predictions.append(pred)

            pbar.update(1)

    return predictions


def parse_answer(answer):
    """Parses the prediction result from the model's response"""
    answer = answer.lower()

    # Extract prediction (Yes/No)
    if "yes" in answer:
        prediction = 1
    elif "no" in answer:
        prediction = 0
    else:
        # Default to non-security related if the answer is ambiguous (can be adjusted)
        prediction = 0

    return prediction


def calculate_metrics(df):
    """Calculates evaluation metrics (TP, FP, TN, FN included)"""
    metrics = {
        "precision": precision_score(df["is_security"], df["prediction"], zero_division=0),
        "accuracy": accuracy_score(df["is_security"], df["prediction"]),
        "f1": f1_score(df["is_security"], df["prediction"], zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(df["is_security"], df["prediction"]).ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Adding confusion matrix elements
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


def print_metrics(metrics):
    """Prints evaluation metrics (TP, FP, TN, FN included)"""
    print("\nFinal Evaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")


def save_final_results(df, metrics):
    """Saves the final results and metrics"""
    # Save the complete results to the output file
    df["model"] = CONFIG["model_path"]
    df.to_json(CONFIG["output_file"], orient="records", lines=True)

    # Save evaluation metrics to a dedicated final metrics file
    with open(CONFIG["final_metrics_file"], "w") as f:
        f.write("Final Experiment Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model: {CONFIG['model_path']}\n")
        f.write(f"Dataset: {CONFIG['input_file']}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['fpr']:.4f}\n")


    print(f"\nFinal metrics saved to {CONFIG['final_metrics_file']}")


def main():
    """Main function"""
    # Load data
    built_prompt = built_baseprompt
    print("Loading data...")
    df = load_data(CONFIG, built_prompt)
    print(f"Loaded {len(df)} samples")

    # Execute prediction
    predictions, confidences = predict_with_Llama(df["prompt"].tolist(), df)
    df["prediction"] = predictions
    df["confidence"] = confidences

    # Calculate final metrics
    print("\nCalculating final metrics...")
    final_metrics = calculate_metrics(df)
    print_metrics(final_metrics)

    # Save final results and metrics
    save_final_results(df, final_metrics)

    # Save the dataframe with predictions and confidence
    df.to_json("llama_Pllm_security_patch_predictions.json", orient="records", lines=True)


if __name__ == "__main__":
    main()