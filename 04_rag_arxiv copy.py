import ssl
import os
import warnings
from dotenv import load_dotenv

# 1. Critical SSL and Warning Fixes
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 2. Fetch Paper (Attention is All You Need: 1706.03762)
print("--- Accessing Arxiv Portal ---")
loader = ArxivLoader(query="1706.03762", load_max_docs=1)
docs = loader.load()

# 3. Chunking & Vector Store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print(f"--- Storing {len(splits)} chunks in ChromaDB ---")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./arxiv_db"
)
retriever = vectorstore.as_retriever()

# 4. LCEL RAG Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

template = """Use the context below to answer the question about the research paper.
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run Query
query = "which year this paper was published and who are the authors?"
print(f"\nQuerying: {query}")
print("-" * 30)
print(rag_chain.invoke(query))