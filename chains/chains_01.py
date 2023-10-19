from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, validator
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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


class Joke(BaseModel):
    setup: str
    punchline: str


parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer user quesry. \n {format_instructions} \n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})


chain = prompt | llm | parser

chain.invoke({"query": "Tell me a joke."})
