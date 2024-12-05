"""TODO:"""

import os
import environment_var
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from fastapi import FastAPI
from openai import OpenAI
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM





class Item(BaseModel):
    """
    TODO:
    """
    query: str
    
    def __init__(self, query:str) -> None:

        super().__init__(query=query)

             
model_name = 'sentence-transformers/msmarco-bert-base-dot-v5'
model_kwargs = {"device": "mps"}  # Device -> cuda:0 or mps
encode_kwargs = {"normalize_embeddings": True}


os.environ["HF_TOKEN"] = environment_var.hf_token
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

use_nvidia_api = False
use_quantized = False

if environment_var.nvidia_key !="":

    client_ai = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=environment_var.nvidia_key
    )

    use_nvidia_api = True

elif use_quantized:

    model_id = "Kameshr/LLAMA-3-Quantized"
    tokenizer =  AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )


# TODO: Get vector client from vector_db class
qdrant = Qdrant(
    client=QdrantClient(path="/Users/egbertoaraujo/WorskSpace/R&D/LLM/generative_search_engine/qdrant"),
    collection_name="LocalCollection",
    embeddings=hf
)



# TODO: API part

app = FastAPI()


@app.get("/")
async def root():
    """
    TODO:
    """

    return {"message": "Hello World"}

@app.post("/search")
def search(item: Item):
    """
    TODO:
    """

    search_result = qdrant.similarity_search(
        query=item.query,
        k=10
    )

    idx_id = 0
    list_response = []

    for response_i in search_result:

        list_response.append(
            {"id": idx_id,
             "path": response_i.metadata.get("path"),
             "content": response_i.page_content
            }
        )
        idx_id += 1

    return list_response

@app.post("/ask_localai")
async def ask_localai(item: Item):
    """
    TODO:
    """

    search_result = qdrant.similarity_search(
        query=item.query,
        k=10
    )

    idx_id = 0
    list_response = []
    context = ""
    mappings = {}

    for response_i in search_result:

        context = context + str(idx_id) + "\n" + response_i.page_content + "\n\n"
        mappings[idx_id] = response_i.metadata.get("path")
        list_response.append(
            {"id": idx_id,
            "path": response_i.metadata.get("path"),
            "context": response_i.page_content
            }
        )

        idx_id += 1

    role_msg = {
        "role": "system",
        "content": "Answer user's question using documents given in the context. In the context are documents that should contain an answer. Please always reference document id (in squere brackets, for example [0],[1]) of the document that was used to make a claim. Use as many citations and documents as it is necessary to answer question."
    }

    messages = [
        role_msg,
        {"role": "user",
         "content": f"Documents:\n{context}\n\nQuestion: {item.query}"},
    ]

    if use_nvidia_api:

        completion = client_ai.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=messages,
            temperature=0.5,
            top_1=1,
            max_tokens=1024,
            stream=False
        )

        response = completion.choices[0].message.content
    else:

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:]
        )
    
    return {"context": list_response, "answer": response}
