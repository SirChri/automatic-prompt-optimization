from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template
from vllm import LLM, SamplingParams
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import utils
import tasks

import os
import gc

######
# GlobalVars
######
file_path = os.path.abspath(os.path.dirname(__file__))

class VLLMPredictor(ABC):
    
    def __init__(self, opt):
        self.opt = opt
        self.llm = None
        self.llm_syn = None

    def eval_multiple(self, prompt, n=1):
        if self.llm_syn is None:
            # This vLLM function resets the global variables, which enables initializing models
            destroy_model_parallel()

            # If you face CUDA OOM Error, then delete all the left over queued operations

            del self.llm
            self.llm = None
            torch.cuda.synchronize() 
            gc.collect()

            self.llm_syn = LLM(model=self.opt["model"] if self.opt["paraphraser"] is None else self.opt["paraphraser"],  download_dir=file_path+"/../cache", gpu_memory_utilization=.8, dtype="half", seed=420)

        outputs = self.llm_syn.generate([prompt for i in range(n)], SamplingParams(temperature=0.8, max_tokens=1024))
        return [output.outputs[0].text for output in outputs]

    def inference(self, ex, prompt):
        if self.llm is None:
            # This vLLM function resets the global variables, which enables initializing models
            destroy_model_parallel()

            # If you face CUDA OOM Error, then delete all the left over queued operations

            del self.llm_syn
            self.llm_syn = None
            torch.cuda.synchronize() 
            gc.collect()

            self.llm = LLM(model=self.opt["model"],  download_dir=file_path+"/../cache", gpu_memory_utilization=.8, dtype="half", seed=420)

        prompt = Template(prompt).render(text=ex['text'])
        res = self.llm.generate([prompt], SamplingParams(temperature=self.opt["temperature"], max_tokens=self.opt["max_tokens"]))
        response = res[0].outputs[0].text
        pred = 1 if response.strip().upper().startswith('REFUTES') else 0
        return pred
