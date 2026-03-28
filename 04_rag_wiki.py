import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Environment & Suppress Pydantic V1/Python 3.14 Warnings
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 2. Load Data from Wikipedia
# This fetches the specific page for the Transformer paper
print("--- Fetching data from Wikipedia ---")
loader = WikipediaLoader(query="Attention is All You Need", load_max_docs=1)
docs = loader.load()

# 3. Chunk the Data
# We break the long article into smaller pieces for the vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 4. Create/Load ChromaDB Vector Store
# This saves the data to the './chroma_db' directory so you can reuse it
print(f"--- Storing {len(splits)} chunks in ChromaDB ---")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()

# 5. Build the RAG Chain using LCEL
llm = ChatOpenAI(model="gpt-4o", temperature=0)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The LCEL Pipe Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Run a Query
response = rag_chain.invoke("can you print the attention is all you need paper?")

print("\n--- Final RAG Response ---")
print(response)