from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import List

class RelevanceChecker:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def check(self, query: str, top_k_chunks: List[str]) -> str:
        
        if not top_k_chunks:
            return "NO_MATCH"
        
        document_content = "\n".join(top_k_chunks)
        
        prompt = ChatPromptTemplate([
            ("system",
             """You are a professional relevance checker between user's query and provided document content.
                **Instructions:**
                - Classify how well the document content addresses the user's query.
                - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
                - Do not include any additional text or explanation.
                **Labels:**
                1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
                2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
                3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
                **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
                **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
             """),
            ("human",
             """**User Query**: {query}\n **Document Content**: {document_content}
             """)
        ])
        
        relevance_chain = prompt | self.llm
        response = relevance_chain.invoke({"query":query, "document_content":document_content})
        #print("Relevance Checker:",response.content)
        return response.content
    