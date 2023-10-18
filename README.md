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
