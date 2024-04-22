from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
import openai
import time
from openai import OpenAIError
from peft import PeftModel

logger = logging.getLogger(__name__)


class AbstractLLMInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def infer_model(self, system_prompt:str, task_prompt:str, in_context_examples: Optional[List]) -> str:
        pass

class HuggingFaceInterface(AbstractLLMInterface):
    def __init__(
        self,
        model_name: str,
        lora_adapter: str,
        system_promt_start_token: str,
        system_promt_end_token: str,
        instruction_start_token: str,
        instruction_end_token: str,
        eos_token: str="",
        bos_token: str="",
    ):
        super().__init__()
        self._model_name = model_name
        self._lora_adapter = lora_adapter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._system_promt_start_token = system_promt_start_token
        self._system_promt_end_token = system_promt_end_token
        self._instruction_start_token = instruction_start_token
        self._instruction_end_token = instruction_end_token
        self._eos_token = eos_token
        self._bos_token = bos_token

    def initialize(self):
        logger.debug("Loading Pretrained Tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=False)
        self._bos_token = self._tokenizer.bos_token
        self._eos_token = self._tokenizer.eos_token
        logger.debug("Loading Tokenizer...DONE")
        logger.debug("Loading Pretrained HuggingFace Model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
        ).eval()
        self._model=self._model.to(self.device)
        logger.debug("Loading Pretrained HuggingFace Model...DONE")
        if self._lora_adapter is None:
            logger.debug("No LoRA adapter provided, skipping LoRA initialization")
        else:
            logger.debug("Loading LoRA Adapter...")
            lora_adapter = PeftModel.from_pretrained(self._model, self._lora_adapter).to(self.device)
            self._model = lora_adapter.merge_and_unload()
            logger.debug("Loading LoRA Adapter...DONE")

    def infer_model(
        self,
        system_prompt:str,
        task_prompt:str,
        in_context_examples: Optional[List],
        max_new_tokens: int=256,
    ) -> str:
        formatted_system_prompt = f"{self._bos_token}{self._instruction_start_token} {self._system_promt_start_token}\n{system_prompt}\n{self._system_promt_end_token}\n"
        formatted_user_prompt = f"{task_prompt} {self._instruction_end_token} "
        if in_context_examples is not None and len(in_context_examples) > 0:
            formatted_examples_prompt = [
                f"{sample[0]} {self._instruction_end_token} {sample[1]} {self._eos_token}{self._bos_token}{self._instruction_start_token} " 
                for sample in in_context_examples
            ]
            formatted_examples_prompt = "".join(formatted_examples_prompt)
            prompt = formatted_system_prompt+formatted_examples_prompt+formatted_user_prompt
        else:
            prompt = formatted_system_prompt+formatted_user_prompt

        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            start_index = inputs["input_ids"].shape[-1]
            output = self._model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=max_new_tokens)
            generation_output = output[0][start_index:]
            ans = self._tokenizer.decode(generation_output, skip_special_tokens=True)
        return ans
    
    def __getstate__(self) -> Dict:
        return {
            "model_name": self._model_name,
            "lora_adapter": self._lora_adapter,
            "system_promt_start_token": self._system_promt_start_token,
            "system_promt_end_token": self._system_promt_end_token,
            "instruction_start_token": self._instruction_start_token,
            "instruction_end_token": self._instruction_end_token,
        }
    
    def __setstate__(self, data: Dict) -> None:
        self.__init__(
            model_name=data["model_name"],
            lora_adapter=data["lora_adapter"],
            system_promt_start_token=data["system_promt_start_token"],
            system_promt_end_token=data["system_promt_end_token"],
            instruction_start_token=data["instruction_start_token"],
            instruction_end_token=data["instruction_end_token"],
        ) 

class OpenAIInterface(AbstractLLMInterface):
    def __init__(
            self,
            model_name: str,
            openai_api_key: str,
            openai_org_key: str,
    ):
        super().__init__()
        self.model_name = model_name
        # we only set these in initialize to support running simulations in parallel with ray
        self.openai_api_key = openai_api_key
        self.openai_org_key = openai_org_key

    def initialize(self):
        openai.api_key = self.openai_api_key

        if self.openai_org_key is not None:
            openai.organization = self.openai_org_key
    
    def infer_model(
        self,
        system_prompt:str,
        task_prompt:str,
        in_context_examples: Optional[List],
        max_retries:int=3,
    ) -> str:
        system_message = [
            {"role": "system","content": system_prompt},
        ]
        task_message = [
            {"role": "user","content": task_prompt}
        ]
        context_messages = []
        if in_context_examples is not None and len(in_context_examples) > 0:
            for sample in in_context_examples:
                context_messages.append(
                    {"role": "user","content": sample[0]}
                )
                context_messages.append(
                    {"role": "assistant","content": sample[1]}
                )
        messages = system_message + context_messages + task_message
        errors = []
        for _ in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                )
                break # successfully infered API
            except OpenAIError as e:
                logger.warning(e)
                logger.warning("Trying again in 0.5s ...")
                errors.append(e)
                time.sleep(0.5)
        else: # failed to infer API
            raise Exception(f"Failed to infer API {max_retries} times with errors {errors}")
        return response["choices"][0]["message"]["content"]