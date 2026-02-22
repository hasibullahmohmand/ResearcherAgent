from typing import List
from langchain_community.retrievers import ArxivRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from pathlib import Path

class ArxivRAGService:
    def __init__(self):
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=4,
            load_all_available_meta=True,
            get_full_documents=True,
            top_k_results=1,
            doc_content_chars_max=2000,
            continue_on_failure=True
        )
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")

    def build_retriever(self,search_queries: List[str]) -> EnsembleRetriever:
        seen_ids = set()
        documents = list()

        for query in search_queries:
            docs = self.arxiv_retriever.invoke(query)
            for doc in docs:
                doc_id = doc.metadata.get("entry_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    documents.append(doc)
                
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        print(len(documents))
        print(seen_ids)
        document_chunks = text_splitter.split_documents(documents)
        
        return self.build_chromadb(document_chunks)
    
    def build_chromadb(self, documents: List[Document], k: int =10, fetch_k: int =10):
        
        chroma = Chroma.from_documents(
            documents=documents,
            embedding=self.embed_model,
            persist_directory=str(Path.cwd() / "temp")
            )
        chroma_retriever = chroma.as_retriever(search_type="mmr", search_kwargs={"k":k, "fetch_k":fetch_k})
        bm25_retriever = BM25Retriever.from_documents(documents, k=k, fetch_k=fetch_k)
        
        return EnsembleRetriever(retrievers=[chroma_retriever,bm25_retriever], weights=[0.4,0.6])    

if __name__ == "__main__":
    r_service = ArxivRAGService()
    search_queries = ["transformer architecture research papers",
                      "transformer models in natural language processing",
                      "academic papers on transformer neural networks"]
    retriever = r_service.build_retriever(search_queries)
    
    docs = retriever.invoke("help me write a topic about transformers.")
    
    print(docs)



