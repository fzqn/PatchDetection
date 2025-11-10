import os
import json
import re
import numpy as np
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
import dataloader

# --- Configuration ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set. Please set it to run the agent.")

CONFIG = {
    "model_name": "deepseek-reasoner",
    "api_base": "https://api.deepseek.com",
    "input_file": "patch_db.json",
    "output_file": "DS_agent_metrics_final.txt",
}

# --- Initialize Model ---
llm = ChatOpenAI(
    model=CONFIG["model_name"],
    openai_api_key=API_KEY,
    openai_api_base=CONFIG["api_base"],
    temperature=0.10,
    max_tokens=512
)

# --- Tool Definitions ---

def get_diff_analysis(diff_content: str) -> str:
    """
    Analyzes *only the added/removed lines* (+/-) from a code diff
    to determine if it's a security patch. Returns Yes/No/Unclear.
    """
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
    response = llm.invoke(prompt).content.strip().lower()
    return "Yes" if "yes" in response else "No" if "no" in response else "Unclear"


def get_diff_message(commit_message: str) -> str:
    """
    Analyzes a commit message to make a *final* 'Yes' or 'No' determination.
    Use this tool *only if* get_diff_analysis returns 'Unclear'.
    """
    prompt = f"""
    Make a final determination based on the *commit message*
    if the commit is a security patch:

    {commit_message}

    You MUST respond with either "Yes" or "No":
    """
    response = llm.invoke(prompt).content.strip().lower()
    return "Yes" if "yes" in response else "No"


# --- Agent Initialization ---

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


system_prompt = """You are an expert software security analyst. Your task is to determine whether a given code commit is a security patch.

You have access to the following tools:

- get_diff_analysis: Analyzes only the added/removed lines (+/-) from a code diff to determine if it's a security patch. Returns 'Yes', 'No', or 'Unclear'.
- get_diff_message: Analyzes the commit message to make a final 'Yes' or 'No' decision. Use this ONLY if get_diff_analysis returns 'Unclear'.

Use the following format:

Question: the input question you must answer.
Thought: [Your step-by-step reasoning here. Consider: What kind of change is this? Does it relate to input validation, memory safety, auth, crypto, etc.?]
Action: the action to take, should be one of [get_diff_analysis, get_diff_message].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat)
Thought: Based on the observation and my reasoning, I now know the answer.
Final Answer: Yes or No.

Important:
- Always perform Chain-of-Thought reasoning in the "Thought" steps.
- Never skip reasoning.
- Your Final Answer must be exactly "Yes" or "No".
"""


# Initialize the ReAct Agent WITHOUT system_message
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)

# --- Prompt Builder ---

def build_react_prompt(row):
    """
    Builds the prompt string (the 'question') for the ReAct agent.
    Injects system_prompt at the beginning.
    """
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


# --- Metrics Functions ---

def calculate_metrics(y_true, y_pred):
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Ensure confusion matrix is 2x2 even with single class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

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
        print("Please update 'input_file' in the CONFIG dictionary.")
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

    output_file = CONFIG["output_file"]
    save_metrics_to_file(metrics, output_file)
    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()
