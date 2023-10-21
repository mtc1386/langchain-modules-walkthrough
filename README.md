# 介绍

本项目旨在通过一系列小 sample 来学习 Langchain，Langchain 作为一个 AI Application Development Framework，其提供的功能主要被分为以下几个 Modules：

- Model I/O
- Retrival
- Chains
- Memory
- Agents
- Callbacks

项目的 sample 会以上述的 Modules 为单元进行整理归纳。

## 准备

LLM 选择的是 \*\*，Langchain 提供 `llama-cpp-python` 来支持与这个 LLM 的交互。开发平台是 MacOS M1

## Model I/O

这个模块提供关于 Model Input 和 Output 操作的接口，包含三个方面：Prompts, Language models, Output parsers.

### Prompts

对于 Model 来说 Input 和 Prompts 意思相同，简单说 Prompts 是用人类的自然语言来描述一个任务让 Model 来完成。不过，如何描述还是很需要技巧和经验的，甚至出现了相应的 Prompt Engineering。

简单的 Prompt 可以是一句明确的指令，比如，回答算数问题：

```
Tell me the result of 10 + 10
```

很多的 Prompt 都可以做成模版方便复用，上面的 Prompt 可以定义为模版 `Tell me the result of {math_quiz}`, 只需每次使用时把 `{math_quiz}` 替换为实际的输入，为此 Langchain 提供了 `PromptTemplate`。

`PromptTemplate` 使用起来和字符串替换类似，先定义字符串模版，并预留占位符，format 的时候提供实际的值。

```py
prompt_template = PromptTemplate.from_template("Tell me the result of {math_quiz}")

prompt = prompt_template.format(math_quiz="10 + 20")
```

此外，PromptTemplate 还允许指定 `input_variables` 来校验，这种情况下，失败会抛出错误。

```py
invalid_prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke about {content}."
)
```

上述内容记录在 `prompt_01.py` 中。

对于更复杂的 Promt Template 需求，Langchain 支持自定义模版，可以继承 `StringPromptTemplate`，用 `pydantic` 来校验 `input_variables`。需要实现方法 `format()->str` 来返回最终的 prompt，可以在该方法里封装更复杂的逻辑。

```py
class CustomPromptTemplate(StringPromptTemplate, BaseModel):
```

上述内容记录在 `prompt_02.py` 中。

#### Few-shot prompt templates

在 Langchain 中提供 `FewShotPromptTemplate` 来处理，few-shot prompt template 包含两个元素 few-shot examples 和 formatter，few-shot examples 可以用 List[dict] 来构造，formatter 必须是 PromptTemplate 对象。

```py
examples = [{"question": "Who lived longer, Muhammad Ali or Alan Turing?","answer":"""
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali"""}]

# prepare the formatter
example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

# feed them into FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)
```

上述内容记录在 `prompt_03.py` 中。

当拥有大量 Examples 时，可能只想选择部分 Example 加入 template 中，你可以选择继承 `BaseExampleSelector` 来实现自定义的 Example Selector，或者选择使用 Langchain 提供的，都放在 `langchain.prompts.example_selector` 下面。

#### Partial prompt template

简单说就是，在 `old template` 只提供部分的 input variables 得到 `new template`。用两个地方可以创建：

1. 调用 `partial()`

```py
origin_prompt_template = PromptTemplate(
    template="{input_1}--{input_2}", input_variables=["input_1", "input_2"])

new_template = origin_prompt_template.partial(input_1="1")

prompt = new_template.format(input_2="2")
```

2. 在 `PromptTemplate` 的构造方法中传入 `partial_variables` 参数，注意这种方式中，在 partial_variables 中提供的值不用被包含在 input_variables 中。

```py
template = PromptTemplate(template="{input_1} -- {input_2}", input_variables=[
    "input_2"], partial_variables={"input_1": "5"})

prompt = template.format(input_2="6")
print(prompt)
```

上面两个方式，除了可以接受字符串的值以外，还可以接受函数，函数的签名需要返回字符串，相关代码记录在 `prompt_04.py` 中。

#### Template Composition

如果想把多个小的 Prompt 组合构建一个大的 Prompt，可以使用 `PipelinePromptTemplate` 。大体分为两步：

1. 定义最终的 Prompt Template，里面的 input variables 会被相应的小的 prompt template 生产的文本替换
2. 把各部分的 Prompt Template 收集起来，数据结构为 List[Tuple]，传递到 `pipeline_prompts` 参数中。

```py
full_template = PromptTemplate.from_template("""\
{first}

{second}

{third}
""")


template_1 = PromptTemplate.from_template("""\
{example}
""")

template_2 = PromptTemplate.from_template("""\
{question}
""")

template_3 = PromptTemplate.from_template("""\
{answer}
""")

prompt_templates = [("first", template_1), ("second",
                                            template_2), ("third", template_3)]

pipeline_template = PipelinePromptTemplate(final_prompt=full_template,
                                           pipeline_prompts=prompt_templates)

prompt = pipeline_template.format(example="this is example",
                                  question="this is question", answer="this is answer")
```

#### Template Serialization

prompt template 可以存储在文件中，这样方便存储，分发，支持两种格式：JSON 和 YAML，Langchain 支持从文件读取 prompt，使用接口 `load_prompt`。
简单的 prompt，需要提供三个字段 `_type`，`input_variables`, `template`

```json
{
  "_type": "prompt",
  "input_variables": ["first", "second"],
  "template": "{first} {second}"
}
```

```py
prompt = load_prompt('prompt.json')
prompt.format(first="F", second="S")
```

除了普通的 prompt，还支持 few-shot prompt, 并且把 example 和 prompt template 存在各自的文件中

```json
[
  { "input": "happy", "output": "sad" },
  { "input": "tail", "output": "head" }
]
```

```json
{
  "_type": "few_shot",
  "input_variables": ["adjective"],
  "prefix": "Write antonyms for the following words.",
  "example_prompt": {
    "_type": "prompt",
    "input_variables": ["input", "output"],
    "template": "Input: {input}\nOutput: {output}"
  },
  "examples": "examples.json",
  "suffix": "Input: {adjective}\nOutput:"
}
```

### Output Parsers

Lang Model 生成的内容是文本，你可以要求文本内容包含结构化的信息，这样方便解析。而 output 的样式是由 prompt 决定的，得通过 prompt 要求 Lang Model 生成想要的内容。

所以，Langchain 提供的 Outparser 一般是这样的工作流程：

1. 定义一个 Outparser。
2. 调用 parser.get_format_instrcutions，填入 prompt 中。
3. 调用 parser.parse(output)，解析结构为想要的结构。

```py
class Joke(BaseModel)
...

parser = PydanticOutputParser(pydantic_object=Joke)
...

prompt_template = PromptTemplate(
    template="Answer user quesry. \n {format_instructions} \n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})
...

output = llm(prompt_template.format(query="Tell me a joke."))
parser.parse(output)

```

上述内容记录在 `outputparsers_01.py`

除了基于 Pydantic 的 OutputParser，Langchain 还提供了几种常见的 OutoutParser，如，CommaSeparatedListOutputParser, DatetimeOutputParser, EnumOutputParser, OutputFixingParser, RetryOutputParser, StructuredOutputParser, XMLOutputParser.

## Chains

在了解 Model I/O 之后，发现交互可以概括为：construct prompt -> call llm -> parse output。 Langchain 的 chain 可以组合这些步骤，当然 chain 能做更多的事，我们先从一个简单的例子，看看 chain 是什么样的。

用 chain 把 Prompt, Llm, Outputparser 组合起来，Langchain 建议用最新的 chain 语法（LangChain Expression Language），同时旧的 Chain interface 仍然支持中。

```py
class Joke(BaseModel)
...

parser = PydanticOutputParser(pydantic_object=Joke)
...

prompt = PromptTemplate(
    template="Answer user quesry. \n {format_instructions} \n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})
...

chain = prompt | llm | parser
chain.invoke(query="Tell me a joke.")

```

相关内容记录在 `chains_01.py` 中。

#### Router Chains

Langchain 提供的 RouterChain 可以从候选的 Chains 中选择一个最合适的，把 input 转发给它，让它来完成任务。

MultiPromptChain 需要三个元素，router_chain 用来选择适合的 chain，destination_chain 是一个 Mapping[str,Chain] 对象，存储着所有可供选择的 Chain，default_chain 如果没有合适的 chain，则默认用这个 chain 来完成任务。

其中涉及到的类有 MultiPromptChain, LLMRouterChain,

```py

... ...
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

... ...
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

... ...

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
```

上述内容记录在 `chains_02.py` 中。

#### Sequential Chains

SimpleSequentialChain 可以把多个 chain 串联在一起，依次执行，执行的 output 是下一个 chain 的 input。

```py

synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True)

```

上述内容记录在 `chains_03.py` 中。

#### Transform Chain

TransformChain 接受一个 function，可对输入进行一些变更并返回，通常也是 SequentialChain 中被使用，比如，只提取一大段文字的前 3 段，然后交由后续的 chain 继续处理。

```py

def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func
)

prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

chain.run(LONG_TEXT)
```

上述内容记录在 `chains_04.py` 中。

## Retrieval

有时，LLM 需要用户指定的数据，但这些数据并不是 LLM 训练集的一部分，要使这些数据能被 LLM 使用，通常会这样做：先收集数据，比如 PDF，Text，社交媒体的文本。在收集到后，会进行一些处理，比如把大文本裁剪为多个更小的 chunk，这个步骤叫做 transform，再然后让 llm 对这些 chunk 生成 Embedding，
Embedding 是 vector 格式的数据来表示一个文本的语义，这样就可以在 vector space 中寻找语义相似的文本，简单说就是在 vector space 中越靠近表示越相似。这些 Embedding 会被存入 vector database，稍后被取出并输入模型进行最后结果的生成。

Langchain 提供的 embeddings 功能都放在 `langchain.embeddings` 里。

根据通常的流程：

1. load data

```py
full_text = open("state_of_the_union.txt", "r").read()
```

2. transform ，比如裁剪文本

```py
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(full_text)
```

3. embed each chunck and inject into db

```py
embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(texts, embeddings)
```

4. retrieve from db

```py
retriver = db.as_retriever()
retriver.invoke(query)
```

## Memory

Memory 在这里的意思是指，能记住之前的与 LLM 交互的信息，比如，chat 类型的应用就需要此特性。Memory System 两个最基本的功能是 Reading 和 Writing，下面结合 Chain 来做进一步解释：

1. Chain 在接受 input 时，会从 Memory 中读取之前的信息加入到 Prompt 中，在交给 LLM 处理。
2. 当 LLM 生成结果时，会把当前的 input 和 output 写入 Memory 中，供下一次交互时获取。

常用的 Memory Class 有 ConversationBufferMemory, ConversationEntityMemory, ConversationKGMemory，等等，都在 `langchain.memory` 下

```py
prompt = PromptTemplate.from_template(template=template)
memory = ConversationBufferMemory(
    memory_key="chat_history", ai_prefix="AI Assistant", human_prefix="New Human")
memory.save_context({"input": "Hi"}, {"output": "what's up"})

conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

conversation({"question": " How are you?"})
conversation({"question": "what's weather today?"})
```

## Agents

Agent 的概念是把 LLM 当作一个推理引擎(Reasoning Engine)来选择下一步行为。Langchain 定义了几个术语来描述这个概念

1. AgentAction 表示要执行的动作，通常是一个函数。
2. AgentFinish 表示 Agent 的工作已结束，需要把结构返回给用户。
3. intermediate_steps 是一个数组，记录之前的 action 以及相应的结果。

Agent 的构建与 Chain 十分类似，不一样的地方在于，你要事先定义好 action，并把它与 llm 绑定。Agent 的运行原理类似在一个 loop 中，检查结果，选择 next action 或者 finish，所以 Langchain 提供 AgentExector 来封装这个运行逻辑。

另外，Langchain 文档提示目前只有通过 OpenAI Function Calling 来创建的 Agent 才是最可靠的。

下面是步骤：

1. 先准备 action，具体即使函数，Langchain 用 tool 来表示

```py
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]
```

2. 构建 prompt，注意要留 `agent_scratchpad` 部分，用于填入之前的 intermediate_steps

```py
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant, but bad at calculating lengths of words."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

3. 绑定 llm 和 tools

```py
from langchain.tools.render import format_tool_to_openai_function
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)
```

4. 创建 agent

```py
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
```

5. Langchain 的 AgentExecutor 帮忙简化我们需要自己写的 agent runtime 的工作

```py
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "how many letters in the word educa?"})
```

### 给 Agent 增加 Memory 功能

需要在 prompt 中添加 Memory 填入的区别，每次运行得出结果后，记录 input 和 output 到 Memory 下次调用时传入。

1. prompt 中需要增加 Memoery Variables 区域

```py
from langchain.prompts import MessagesPlaceholder

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant, but bad at calculating lengths of words."),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

2. 每次执行后，记录 input 和 output，并在下次执行时传入

```py
from langchain.schema.messages import HumanMessage, AIMessage
chat_history = []

input1 = "how many letters in the word educa?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.append(HumanMessage(content=input1))
chat_history.append(AIMessage(content=result['output']))
agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})
```

## Callback

Langchain 的 Callback System 可以在 LLM 运行的不同阶段回调相应的方法，最常用的是 logging，比如，把运行日志输出到控制台，可以用 `StdOutCallbackHandler`。

```py
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

# Constructor callback: First, let's explicitly set the StdOutCallbackHandler when initializing our chain
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.run(number=2)
```
