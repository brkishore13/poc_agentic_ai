import ssl
import os
import asyncio
import logging
import warnings
from typing import List
from dotenv import load_dotenv

# --- CRITICAL FIXES FOR PRODUCTION START ---
# 1. Bypass SSL verification for Arxiv downloads (Must be at the top)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Suppress Python 3.14 / Pydantic V1 warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# --- CRITICAL FIXES FOR PRODUCTION END ---

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductionRAG")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.vectorstore = None

    async def ingest_paper(self, arxiv_id: str):
        """Asynchronous Ingestion Pipeline"""
        logger.info(f"Starting ingestion for {arxiv_id}...")
        
        # Load paper
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
        # Running in thread to keep the event loop free
        docs = await asyncio.to_thread(loader.load)
        
        if not docs:
            logger.error(f"No documents found for Arxiv ID: {arxiv_id}")
            return

        # Semantic Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        
        # In production, we use a distinct directory for the vector store
        self.vectorstore = await asyncio.to_thread(
            Chroma.from_documents,
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./arxiv_db_prod"
        )
        logger.info("Ingestion complete and stored in ./arxiv_db_prod")

    def get_chain(self):
        """Optimized LCEL Chain"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_template("""
        Answer the research question using the provided context.
        Context: {context}
        Question: {question}
        Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

async def main():
    rag = RAGSystem()
    
    # Ingest the Transformer paper
    await rag.ingest_paper("1706.03762")
    
    # Run a test query
    chain = rag.get_chain()
    query = "What is the primary benefit of the Transformer architecture over RNNs?"
    
    print(f"\nQuery: {query}\n" + "-"*30)
    result = await chain.ainvoke(query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())