import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np

import tiktoken
import openai
from openai.types import Batch

from collections.abc import Iterable
from utils.env import openai_key
import requests
import json
import re


def get_gpt_tokens(text, model: str) -> Iterable:
    encoding = tiktoken.encoding_for_model(model)
    tokens   = encoding.encode(text)

    return tokens

def get_client():
    global openai_key
    return openai.OpenAI(api_key=openai_key)

def generate_jsonl_file(file_name: str, iterable: Iterable):
    with open(f'{file_name}.jsonl', 'w') as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')

def upload_file_to_openai(file_path: str, purpose: str) -> str:
    client = get_client()
    
    file_id = client.files.create(
      file=open(file_path, "rb"),
      purpose=purpose
    ).id

    return file_id

########################## Batch Processing Util Functions ##########################

def get_batch_completion_tasks(df: pd.DataFrame, user_prompt_col: str, model: str, temperature: float, system_prompt: str, max_tokens: int):
    tasks = []
    for index, row in df.iterrows():
        user_prompt = row[user_prompt_col]
        
        task = {
            "custom_id": f"{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "max_tokens": max_tokens
            }
        }
        
        tasks.append(task)

    return tasks

def create_batch_completion_job(batch_file_id: str, endpoint: str, metadata: dict = {}) -> dict:
    client = get_client()

    return client.batches.create(
        input_file_id=batch_file_id,
        endpoint=endpoint,
        metadata=metadata,
        completion_window="24h"
    )    

def get_batch_job_status(batch_job_id: str) -> Batch:
    client = get_client()
    status = client.batches.retrieve(batch_job_id)
    return status

def get_batch_job_results(file_id: str):
    client = get_client()

    return client.files.content(file_id)

def save_batch_results_to_jsonl(output_file, file_name: str):
    with open(f'{file_name}.jsonl', 'w') as f:
        f.write(output_file.text)

def map_jsonl_batch_completion_results_to_df(df: pd.DataFrame, file_name: str, target_col: str):
    with open(f'{file_name}.jsonl', 'r') as f:
        for line in f:
            try:
                row = json.loads(line)
                idx = int(row['custom_id'])
                res = row['response']['body']['choices'][0]['message']['content']

                df.at[idx, target_col] = res
            except Exception as e:
                print(f'Error reading line: {e}\n', f'{line}\n', '_'*40)
        

########################## Fine Tuning Util Functions ##########################

def generate_openai_fine_tuning_json_from_df(fine_tuning_file_name: str, df: pd.DataFrame, system_prompt: str, user_promt_col: str, target_col: str, samples_per_class: int, random_state: int):
    
    def set_row_to_fine_tuning_format(system_prompt: str, user_prompt: str, classification: str) -> dict:    
        return {
            "messages": [
                {"role": "system", "content": fr'''{system_prompt}'''},
                {"role": "user", "content": fr'''{user_prompt}'''},
                {"role": "assistant", "content": fr'''{classification}'''}
            ]
        }    

    
    # Randomly extract sample per target class for a balanced training set
    df_fine_tuning = df[[user_promt_col, target_col]].groupby(target_col).sample(n=samples_per_class, random_state=random_state)

    # Generate the JSONL file
    with open(f'{fine_tuning_file_name}.jsonl', 'w') as f:
        for index, record in df_fine_tuning.iterrows():
            row = set_row_to_fine_tuning_format(system_prompt, record[user_promt_col], record[target_col])
            f.write(json.dumps(row) + '\n')

def upload_fine_tuning_file_to_openai(file_path: str) -> str:
    client = get_client()
    
    file_id = client.files.create(
      file=open(file_path, "rb"),
      purpose='fine-tune'
    ).id

    return file_id   
    
def fine_tune_gpt_model(file_id: str, model: str, hyperparameters: dict = {}) -> str:
    '''Creates a Fine Tuning job in OpenAI based on the file and model specified. Returns a job ID'''
    client = get_client()
    job = client.fine_tuning.jobs.create(training_file=file_id, model=model, hyperparameters=hyperparameters)

    return job.id

def check_fine_tuning_job_status(job_id: str, limit: int = 10) -> list:
    client = get_client()
    
    status_events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
    return status_events

def get_fine_tuned_model_details(job_id: str) -> str:
    client = get_client()
    model_name_pre_object = client.fine_tuning.jobs.retrieve(job_id)
    return model_name_pre_object