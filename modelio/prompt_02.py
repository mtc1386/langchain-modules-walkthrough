from langchain.prompts import StringPromptTemplate
from langchain.pydantic_v1 import (validator, BaseModel)
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from abc import ABC
from typing import List
import os


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 1
n_batch = 512
model_path = os.getenv('LLM_MODEL')

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

PROMPT = """\
Tell me the result of the following math quiz:
{math_quiz}
"""


class MathQuizPromptTemplate(StringPromptTemplate, BaseModel):

    @validator("input_variables")
    def validate_input_values(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "math_quiz" not in v:
            raise ValueError("math_quiz must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        prompt = PROMPT.format(math_quiz=kwargs['math_quiz'])
        return prompt

    def _prompt_type(self) -> str:
        return "math helper"


prompt_template = MathQuizPromptTemplate(input_variables=["math_quiz"])

prompt = prompt_template.format(math_quiz="10 + 10")

llm(prompt)
