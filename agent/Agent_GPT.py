import os
import json
import re
import numpy as np
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import dataloader

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to run the agent.")

CONFIG = {
    "model_name": "gpt-5",
    "input_file": "patch_db.json",
    "output_file": "GPT5_agent_metrics_final.txt",
}

# --- Initialize Model ---
llm = ChatOpenAI(
    model=CONFIG["model_name"],
    openai_api_key=API_KEY,
    temperature=0.10,
    max_tokens=512
)

# --- Tool Definitions ---

def get_diff_analysis(diff_content: str) -> str:
    changed_lines = []
    for line in diff_content.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            changed_lines.append(line)
        elif line.startswith('-') and not line.startswith('---'):
            changed_lines.append(line)

    if not changed_lines:
        return "Unclear"

    changed_diff = "\n".join(changed_lines)

    prompt = f"""
    Carefully analyze the following *code changes* (only lines starting with + or - are shown)
    to determine if this commit is a security patch:

    {changed_diff}

    Respond strictly with:
    1. "Yes" if clearly a security fix
    2. "No" if clearly non-security related
    3. "Unclear" if uncertain
    """
    try:
        response = llm.invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"\nDiff analysis error: {str(e)}")
        return "Unclear"

    if "yes" in response: return "Yes"
    if "no" in response: return "No"
    return "Unclear"


def get_diff_message(commit_message: str) -> str:
    prompt = f"""Make a final determination based on the *commit message*
    if the commit is a security patch:

    {commit_message}

    You MUST respond with either "Yes" or "No":
    """
    try:
        response = llm.invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"\nMessage analysis error: {str(e)}")
        return "No"

    return "Yes" if "yes" in response else "No"


# --- Tools ---
tools = [
    Tool(
        name="get_diff_analysis",
        func=get_diff_analysis,
        description="Analyzes code changes (only +/- lines) to determine if it's a security patch. Returns 'Yes', 'No', or 'Unclear'. Use this first."
    ),
    Tool(
        name="get_diff_message",
        func=get_diff_message,
        description="Analyzes commit message to make a final 'Yes' or 'No' decision. Only use if get_diff_analysis returns 'Unclear'."
    )
]

# --- System Prompt (to be manually prepended) ---
system_prompt = """You are an expert software security analyst. Your task is to determine whether a given code commit is a security patch.

You have access to the following tools:

- get_diff_analysis: Analyzes only the added/removed lines (+/-) from a code diff to determine if it's a security patch. Returns 'Yes', 'No', or 'Unclear'.
- get_diff_message: Analyzes the commit message to make a final 'Yes' or 'No' decision. Use this ONLY if get_diff_analysis returns 'Unclear'.

Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [get_diff_analysis, get_diff_message].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Your final answer must be either "Yes" or "No".
"""


# --- Initialize Agent WITHOUT system_message ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)


# --- Prompt Builder: prepend system_prompt manually ---
def build_react_prompt(row):
    diff_code = str(row['diff_code'])
    # Note: commit_message is NOT included here because it's only used conditionally via tool
    return f"""{system_prompt}

Question: Is the following commit a security patch?
Your final answer must be either "Yes" or "No".

### Data for get_diff_analysis (Code Diff) ###
{diff_code}
"""


# --- Metrics Functions (unchanged) ---
def calculate_metrics(y_true, y_pred):
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        metrics["fpr"] = 0.0
    return metrics

def print_metrics(metrics):
    print("\nEvaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f}")

def save_metrics_to_file(metrics, file_path):
    with open(file_path, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['fpr']:.4f}\n")


# --- Main Execution ---
def main():
    try:
        print(f"Loading data from {CONFIG['input_file']}...")
        df = dataloader.load_data(CONFIG, build_react_prompt)
        print(f"Loaded {len(df)} items.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {CONFIG['input_file']}.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    y_true = []
    y_pred = []
    progress_bar = tqdm(total=len(df), desc="Processing commits", unit="commit")

    for i, row in df.iterrows():
        question = row['prompt']
        try:
            result = agent.run(question).strip().lower()
        except Exception as e:
            print(f"\nAgent processing error on item {i + 1}: {str(e)}")
            result = "no"

        final_decision = "yes" if "yes" in result else "no"
        true_label_str = "SEC" if row["is_security"] == 1 else "NON"

        y_true.append(row["is_security"])
        y_pred.append(1 if final_decision == "yes" else 0)

        progress_bar.update(1)
        progress_bar.set_postfix({"Result": final_decision.upper(), "True": true_label_str})

    progress_bar.close()

    if not y_true:
        print("No data was processed. Exiting.")
        return

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    save_metrics_to_file(metrics, CONFIG["output_file"])
    print(f"\nMetrics saved to {CONFIG['output_file']}")

    # Confusion matrix
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("\nConfusion Matrix:")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
    else:
        print("\nConfusion Matrix: Cannot be calculated. Only one class present in true labels.")


if __name__ == "__main__":
    main()