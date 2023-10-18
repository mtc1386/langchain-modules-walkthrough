from langchain.prompts import PromptTemplate


origin_prompt_template = PromptTemplate(
    template="{input_1}--{input_2}", input_variables=["input_1", "input_2"])


new_template = origin_prompt_template.partial(input_1="1")

prompt = new_template.format(input_2="2")

print(prompt)

# Notice: input_variable doesn't contain input_1
template = PromptTemplate(template="{input_1} -- {input_2}", input_variables=[
    "input_2"], partial_variables={"input_1": "5"})

prompt = template.format(input_2="6")
print(prompt)


# Using function that return str
template_3 = PromptTemplate(
    template="{input_1} -- {input_2}", input_variables=["input_1", "input_2"])
prompt_3 = template_3.partial(input_1=lambda: "9")
print(prompt_3.format(input_2="10"))


template_4 = PromptTemplate(template="{input_1} -- {input_2}", input_variables=[
                            "input_2"], partial_variables={"input_1": lambda: "00"})
prompt_4 = template_4.format(input_2="11")
print(prompt_4)
