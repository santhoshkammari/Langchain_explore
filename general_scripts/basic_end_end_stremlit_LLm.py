import streamlit as st

llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_reply = llm_chain.run(myprompt)
reply = llm_reply.partition('<|end|>')[0]

from transformers import AutoModelForCausalLM, AutoTokenizer

model = "HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(repo_id=model ,
                         model_kwargs={"min_length":30,
                                       "max_new_tokens":256, "do_sample":True,
                                       "temperature":0.2, "top_k":50,
                                       "top_p":0.95, "eos_token_id":49155})

st.title("AI assistant")
user_input = st.text_input("You:", "")
if st.button("Send"):
    response = lang_chain.generate_response(user_input)
    st.text_area("Chatbot:", response, height=100)