import re
import numpy as np
import pandas as pd

import torch
#from torch.utils.data import Dataset, DataLoader
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel

from trl import SFTTrainer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc


def remove_special_chars(text):
    return re.sub(r"[^\w\s]", "", text)

# Prepare the prompts using Dataset.map with batched=True
def prepare_prompts(examples, text_column, system_prompt, tokenizer):
    messages_list = [
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": text},
        ]
        for text in examples[text_column]
    ]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    return {"prompts": prompts}


def process_llm_on_dataset_batches(df: pd.DataFrame, text_column: str, system_prompt: str, model_name: str, tokenizer_model: str, temperature=0.7, max_new_tokens=256, batch_size=16):
    try:
        # Load tokenizer with trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        
        # Initialize the pipeline with batch_size
        pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            batch_size=batch_size
        )
      
        # Convert DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[[text_column]])
        dataset = dataset.map(lambda x: prepare_prompts(x, text_column, system_prompt, tokenizer), batched=True)
        
        # Process the prompts using the pipeline
        outputs = pipe(
            dataset["prompts"],
            max_new_tokens=max_new_tokens,
            do_sample=(temperature != 0.0),
            temperature=temperature if temperature != 0.0 else None,
        )
        
        # Extract the generated texts
        responses = [output[0]['generated_text'] for output in outputs]
    
        # Clean up
        del pipe, outputs, tokenizer, dataset
        torch.cuda.empty_cache()
        
        return responses
            
    except Exception as e:
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()
        raise e


def collate_fn(batch):
    input_texts, output_texts = zip(*batch)
    return list(input_texts), list(output_texts)


def prepare_train_data(df: pd.DataFrame, text_col: str, target_col: str, system_prompt: str):
    df = df.copy()
    df["text"] = df[[text_col, target_col]].apply(lambda x: "<|system|>\n" + system_prompt + "</s>\n" + "<|user|>\n" + x[text_col] + "</s>\n<|assistant|>\n" + x[target_col] + "</s>", axis=1)

    df = df.sample(frac=1, random_state=12345).reset_index(drop=True)

    # Split to 80/20
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    eval_df = df.iloc[train_size:]

    train_data = Dataset.from_pandas(train_df)
    eval_data  = Dataset.from_pandas(eval_df)
    return train_data, eval_data


def get_quantized_model_and_tokenizer(mode_id):

    tokenizer = AutoTokenizer.from_pretrained(mode_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        mode_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=False
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer