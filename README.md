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

在 Langchain 中提供 `FewShotPromptTemplate` 来处理，few-shot prompt template 包含两个元素 few-shot examples 和 formatter，few-shot examples 可以用 dict 来构造，formatter 必须是 PromptTemplate 对象。

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
