from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
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


template = """ You are a nice chat bot and you are having a conversation with a human.

Previous conversation:
{chat_history}

New Human: {question}
AI Assistant: ""
"""

prompt = PromptTemplate.from_template(template=template)
memory = ConversationBufferMemory(
    memory_key="chat_history", ai_prefix="AI Assistant", human_prefix="New Human")
memory.save_context({"input": "Hi"}, {"output": "what's up"})

conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

conversation({"question": " How are you?"})
conversation({"question": "what's weather today?"})
