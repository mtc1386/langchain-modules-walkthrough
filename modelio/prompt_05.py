from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate


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

#
prompt_templates = [("first", template_1), ("second",
                                            template_2), ("third", template_3)]

pipeline_template = PipelinePromptTemplate(final_prompt=full_template,
                                           pipeline_prompts=prompt_templates)

prompt = pipeline_template.format(example="this is example",
                                  question="this is question", answer="this is answer")


print(prompt)
