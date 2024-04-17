import json
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# print(docs[0])
texts = docs[0].page_content.strip()
embedder = SpacyEmbeddings(model_name="en_core_web_sm")
# embeddings = embedder.embed_documents(texts)
# for i, embedding in enumerate(embeddings):
#     print(f"Embedding for document {i+1}: {embedding}")

for doc in docs:
    doc.metadata["source"]=os.path.basename(doc.metadata['source'])
    vector_db = Milvus.from_documents(
        docs,
        embedder,
        drop_old = True,
        connection_args={"host": os.getenv("MILVUS_HOST"), "port": os.getenv("MILVUS_PORT")},
    )

query = "Where did Jack go"
docs = vector_db.similarity_search_with_score(query, k=3)

response = []
for doc, score in docs:
    response.append({
        "page_content": doc.page_content.strip(),
        "score": score,
        "metadata": doc.metadata['source']
    })

print(json.dumps(response, indent=4))

# print(docs[0].page_content)


