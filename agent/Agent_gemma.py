import os
import json
import re
import numpy as np
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
import dataloader

# --- Configuration ---
CONFIG = {
    "model_path": "YOUR/LOCAL/GEMMA3/PATH",
    "input_file": "patch_db.json",
    "output_file": "gamma_agent_metrics.txt",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "context_window": 8192,
}

# --- Initialize Gemma Model ---
print("Loading local Gemma-3-7B model...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_path"],
    torch_dtype=torch.float16,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=CONFIG["max_new_tokens"],
    temperature=CONFIG["temperature"],
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Tool Definitions (Gemma chat format) ---

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

    prompt = f"""<start_of_turn>user
Carefully analyze the following *code changes* (only lines starting with + or - are shown) to determine if this commit is a security patch:

{changed_diff}

Respond strictly with:
1. "Yes" if clearly a security fix
2. "No" if clearly non-security related
3. "Unclear" if uncertain<end_of_turn>
<start_of_turn>model
"""

    response = llm.invoke(prompt).strip().lower()
    if "yes" in response:
        return "Yes"
    if "no" in response:
        return "No"
    return "Unclear"


def get_diff_message(commit_message: str) -> str:
    prompt = f"""<start_of_turn>user
Make a final determination based on the *commit message* if the commit is a security patch:

{commit_message}

You MUST respond with either "Yes" or "No".<end_of_turn>
<start_of_turn>model
"""

    response = llm.invoke(prompt).strip().lower()
    return "Yes" if "yes" in response else "No"


# --- Agent Tools ---
tools = [
    Tool(
        name="get_diff_analysis",
        func=get_diff_analysis,
        description="Get specific code diff information. Analyzes only added/removed lines to determine if it's a security patch. Returns 'Yes', 'No', or 'Unclear'."
    ),
    Tool(
        name="get_diff_message",
        func=get_diff_message,
        description="Get additional commit information. Analyzes commit message to make a final 'Yes' or 'No' decision. Only use if get_diff_analysis returns 'Unclear'."
    )
]

# --- System Prompt (will be manually prepended to each question) ---
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
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)


# --- Prompt Builder: prepend system_prompt manually ---
def build_react_prompt(row):
    diff_code = str(row['diff_code'])
    commit_message = str(row['commit_message'])

    return f"""{system_prompt}

Question: Is the following commit a security patch?
Your final answer must be either "Yes" or "No".

Here is the data for your tools:

### Data for get_diff_analysis (Code Diff) ###
{diff_code}

### Data for get_diff_message (Commit Message) ###
{commit_message}
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

    for i, row in df.iterrows():
        print(f"\n--- Processing item {i + 1}/{len(df)} (Index: {row.name}) ---")
        question = row['prompt']

        try:
            result = agent.run(question).strip().lower()
        except Exception as e:
            print(f"Agent processing error: {str(e)}")
            result = "no"

        final_decision = "yes" if "yes" in result else "no"
        true_label_str = "security" if row["is_security"] == 1 else "non-security"
        print(f"Ground Truth: {true_label_str} -> Final Decision: {final_decision}")

        y_true.append(row["is_security"])
        y_pred.append(1 if final_decision == "yes" else 0)

    if not y_true:
        print("No data was processed. Exiting.")
        return

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    save_metrics_to_file(metrics, CONFIG["output_file"])
    print(f"\nMetrics saved to {CONFIG['output_file']}")


if __name__ == "__main__":
    main()