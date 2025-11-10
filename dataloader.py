import pandas as pd
import json

def load_data(config ,prompt_type):
    with open(config["input_file"], "r",encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 生成标签
    df["is_security"] = df["category"].apply(lambda x: 1 if x == "security" else 0)

    build_prompt = prompt_type
    df["prompt"] = df.apply(build_prompt, axis=1)
    return df