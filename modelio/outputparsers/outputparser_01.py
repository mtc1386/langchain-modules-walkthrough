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

    @validator("setup")
    def setup_should_end_with_question_mark(cls, field):
        if field[-1] != '?':
            raise ValueError(
                "setup should be a question end with question mark")
        return field


parser = PydanticOutputParser(pydantic_object=Joke)

prompt_template = PromptTemplate(
    template="Answer user quesry. \n {format_instructions} \n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})


output = llm(prompt_template.format(query="Tell me a joke."))
print(f'output: {output}')
joke = parser.parse(output)

print(joke)
