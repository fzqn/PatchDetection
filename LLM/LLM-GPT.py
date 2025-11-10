import openai
import time
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from dataloader import *
from prompt import *
import requests
import json
import os

# Configuration settings
CONFIG = {
    "api_key": "YOUR_API_KEY_HERE", # Replace with your actual API key
    "api_endpoint": "YOUR_API_ENDPOINT_HERE", # Replace with your actual API endpoint
    "model_name": "gpt-4o",   # Model to call
    "input_file": "patch_db.json",
    "output_file": "chat4-security_patch_predictions.json",
    "metrics_file": "chat4-metrics_new.txt",
    "batch_size": 500,  # Retained but not used for batch metric calculation
    "retry_limit": 3,
    "request_interval": 5  # Seconds
}




def predict_with_openai(prompts, df):
    """Calls the OpenAI API for batch prediction"""
    predictions = []

    with tqdm(total=len(prompts), desc="Predicting") as pbar:
        for i, prompt in enumerate(prompts):
            headers = {
                "Authorization": f"Bearer {CONFIG['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": CONFIG["model_name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
            }

            prediction = 1  # Default prediction is non-security related
            for retry in range(CONFIG["retry_limit"]):
                try:
                    response = requests.post(
                        CONFIG["api_endpoint"],
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"].strip().lower()
                    prediction = parse_answer(answer)  # Get the prediction result directly
                    print(f"API Response: {answer}")
                    # Modified output message for clarity
                    print(f"Prediction: {'Security-related (1)' if prediction == 1 else 'Non-security-related (0)'}")
                    break  # Exit retry loop on success
                except Exception as e:
                    if retry == CONFIG["retry_limit"] - 1:
                        print(f"Failed to process sample {i}: {str(e)}")
                    time.sleep(15)  # Longer interval for retries

            predictions.append(prediction)

            time.sleep(CONFIG["request_interval"])
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
    """Calculates evaluation metrics"""
    metrics = {
        "precision": precision_score(df["is_security"], df["prediction"], zero_division=0),
        "accuracy": accuracy_score(df["is_security"], df["prediction"]),
        "f1": f1_score(df["is_security"], df["prediction"], zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(df["is_security"], df["prediction"]).ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


def print_metrics(metrics):
    """Prints evaluation metrics"""
    print("\nEvaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")


def save_final_results(df, metrics):
    """Saves the final results and metrics"""
    # Save the complete results
    df["model"] = CONFIG["model_name"]
    df.to_json(CONFIG["output_file"], orient="records", lines=True)

    # Save evaluation metrics to a dedicated metrics file
    with open(CONFIG["metrics_file"], "w") as f:
        f.write("Experiment Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Dataset: {CONFIG['input_file']}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['fpr']:.4f}\n")

    print(f"\nFinal results saved to {CONFIG['metrics_file']}")


def main():
    """Main function"""
    # Execute prediction
    built_prompt = built_baseprompt
    print("Loading data...")
    df = load_data(CONFIG, built_prompt)
    predictions = predict_with_openai(df["prompt"].tolist(), df)
    df["prediction"] = predictions
    df = df[df["prediction"] != -1]  # Filter out failed requests

    # Calculate final metrics
    print("\nCalculating final metrics...")
    final_metrics = calculate_metrics(df)
    print("\nFinal Evaluation Results:")
    print_metrics(final_metrics)

    # Save final metrics
    save_final_results(df, final_metrics)


if __name__ == "__main__":
    main()