from langchain_community.llms.huggingface_hub import HuggingFaceHub
from transformers import AutoModelForCausalLM, AutoTokenizer

model = "openai-community/gpt2"
llm = HuggingFaceHub(repo_id=model ,
huggingfacehub_api_token = "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",
                         model_kwargs={"min_length":30,
                                       "max_new_tokens":250, "do_sample":True,
                                       "temperature":0.2,
                                       "top_p":0.95, "eos_token_id":49155})
print(llm.invoke("write a python script to sum 1 to 10 numbers"))

