from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
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


synopsis_template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
synopsis_prompt_template = PromptTemplate(
    input_variables=["title"], template=synopsis_template)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(
    input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True)


chain.run("Tragedy at sunset on the beach")
