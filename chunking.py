from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json


def create_docs(data):
    embed_fun = HuggingFaceEmbeddings(model_name = "deepvk/USER-bge-m3")
    text_splitter = SemanticChunker(embed_fun,breakpoint_threshold_type="interquartile")
    texts = text_splitter.create_documents([data["text"]],)
    return texts


if __name__=="__main__":
    with open("output212.json", "r") as f:
        docs = json.load(f)
    l = []
    for i in docs.keys():
        if "text" in  docs[i]:
            splits = create_docs(docs[i])
            l+=splits
    print(len(l))
    with open("chunks_q_212.txt","w+") as f:
        for i in l:
            f.write(i.page_content+"\n\n")

