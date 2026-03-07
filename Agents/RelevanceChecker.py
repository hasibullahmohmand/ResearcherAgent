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
                - Always respond with exactly one label: CAN_ANSWER, PARTIAL, or NO_MATCH.
                - Do NOT add punctuation, quotes, or extra explanation.
                - Only classify based on the provided document content.
                **Labels:**
                1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
                2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
                3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
                **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
                **RESPOND ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
             """),
            ("human",
             """**User Query**: {query}\n **Document Content**: {document_content}
                RESPOND ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH
             """)
        ])
        
        relevance_chain = prompt | self.llm
        response = relevance_chain.invoke({"query":query, "document_content":document_content})
        print("Relevance Checker:",response.content)
        if response.content not in ["CAN_ANSWER", "PARTIAL", "NO_MATCH"]:
            return "NO_MATCH"
        return response.content
    
if __name__ == "__main__":
    content = """The concepts of Gross Domestic Product (GDP), GDP per capita, and population are central to
    the study of political science and economics. However, a growing literature suggests that existing 
    measures of these concepts contain considerable error or are based on overly simplistic modeling choices.
    We address these problems by creating a dynamic, three-dimensional latent trait model, which uses observed
    information about GDP, GDP per capita, and population to estimate posterior prediction intervals for each 
    of these important concepts. By combining historical and contemporary sources of information, we are able 
    to extend the temporal and spatial coverage of existing datasets for country-year units back to 1500 A.D 
    through 2015 A.D. and, because the model makes use of multiple indicators of the underlying concepts, we 
    are able to estimate the relative precision of the different country-year estimates. Overall, our latent 
    variable model offers a principled method for incorporating information from different historic and 
    contemporary data sources. It can be expanded or refined as researchers discover new or alternative 
    sources of information about these concepts."""
    user_query = "How do historical trauma and socio-economic impacts of the Afghanistan conflict influence contemporary governance structures?"
    
    r =RelevanceChecker()
    r.check(user_query,[content])