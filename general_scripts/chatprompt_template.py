from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoTokenizer, AutoModel, pipeline, Pipeline,AutoModelForCausalLM
StrOutputParser
def chat_template():
    template = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("user", "{user}")
    ])

    # prompt_value = template.invoke("Hello, there!")
    # print(prompt_value)

    messages = template.invoke({"system":"Hello, there",
                                "user":"How are you?"
                                })

    ans =""
    for message in messages.messages:
        if isinstance(message,SystemMessage):
          ans += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{message.content}<|eot_id|>"
        elif isinstance(message,HumanMessage):
            ans+=f"<|start_header_id|>user<|end_header_id|>\n{message.content}<|eot_id|>"
    ans+=f"<|start_header_id|>assistant<|end_header_id|>"
    print(ans)

    print(messages.messages[0].content)

def tinyllama():
    tokenz= AutoTokenizer.from_pretrained("/home/ntlpt59/MAIN/models/tinyllama_tokenizer")
    model = AutoModelForCausalLM.from_pretrained("/home/ntlpt59/MAIN/models/tinyllama_model")
    pipe: Pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenz
    )
    messages = [
        {"role": "system", "content": "You are bot"},
        {"role": "user", "content": "What is your name"},
        {"role":"assistant","content": "hi,my name is isha"},
        {"role": "user", "content": "how old are you?"},
        {"role": "assistant", "content": "no idea"},
        {"role": "user", "content": "i forgot your name, what was it?"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
    ]

    outputs = pipe(
        prompt,
        max_new_tokens=50,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])


if __name__ == '__main__':
    tinyllama()