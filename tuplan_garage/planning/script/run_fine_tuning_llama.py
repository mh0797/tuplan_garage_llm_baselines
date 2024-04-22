import os
import hydra
from hydra.utils import instantiate
import torch
import json
from datasets import Dataset
import pandas as pd
from omegaconf import DictConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def parse_dataset(
        dataset_path: str,
        bos_token: str="<s>",
        eos_token: str="</s>",
        instruction_start_token: str="[INST]",
        instruction_end_token: str="[/INST]",
        system_promt_start_token: str="<<SYS>>",
        system_promt_end_token: str="<</SYS>>",
    ) -> Dataset: 
    with open(dataset_path, "r") as f:
        dataset = list(f)
    dataset = [json.loads(sample)["messages"] for sample in dataset]

    df_items = []
    for sample in dataset:
        system_messages = [message["content"] for message in sample if message["role"] == "system"]
        user_messages = [message["content"] for message in sample if message["role"] == "user"]
        assistant_messages = [message["content"] for message in sample if message["role"] == "assistant"]
        assert len(system_messages) == len(assistant_messages) == len(user_messages) == 1

        system_prompt = system_messages[0]
        task_prompt = user_messages[0]
        assistant_response = assistant_messages[0]

        formatted_system_prompt = f"{bos_token}{instruction_start_token} {system_promt_start_token}\n{system_prompt}\n{system_promt_end_token}\n"
        formatted_user_prompt = f"{task_prompt} {instruction_end_token} "
        formatted_assistant_response = f"{assistant_response} {eos_token}"

        item = formatted_system_prompt + formatted_user_prompt + formatted_assistant_response

        df_items.append(
            pd.DataFrame([[item]], columns=["text"])
        )
    dataset = pd.concat(df_items, axis=0).reset_index(drop=True)
    return Dataset.from_pandas(dataset)

@hydra.main(config_path="../script/config/fine_tuning", config_name="default_fine_tuning")
def main(cfg: DictConfig):
    print("Output directory is",cfg.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    dataset = parse_dataset(cfg.dataset)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    lora_config: LoraConfig = instantiate(cfg.lora_config)
    training_params: TrainingArguments = instantiate(cfg.training_arguments)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()

    model_save_path = os.path.join(cfg.output_dir, "ft_"+cfg.model_name)
    trainer.save_model(model_save_path)

if __name__=="__main__":
    main()