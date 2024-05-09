from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def load_hf_llm():
    hf = HuggingFacePipeline.from_model_id(
        model_id = "deepset/roberta-base-squad2",
        task = "text-generation",
        pipeline_kwargs = {
                        "max_new_tokens":10
                          }
    )
    return hf

if __name__ == '__main__':
    hf = load_hf_llm()
    response = hf.invoke(
        input="hai"
    )
    print(response)