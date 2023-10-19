from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
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


LONG_TEXT = """
Introduction

LangChain is a framework for developing applications powered by language models. It enables applications that:

    Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
    Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

The main value props of LangChain are:

    Components: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
    Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks

Off-the-shelf chains make it easy to get started. For complex applications, components make it easy to customize existing chains and build new ones.
"""


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func
)

template = """Summarize this text:

{output_text}

Summary:"""
prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

chain.run(LONG_TEXT)
