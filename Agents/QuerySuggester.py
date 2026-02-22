from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class Output(BaseModel):
    search_queries: List[str] = Field("3 serach queries for finding relevant academic papers.")

class QuerySuggester:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
    def research(self, user_query: str) -> Output:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a professional researcher.
             Analyze the user's query and suggest 3 search queries for finding relevant academic papers.
             """),
            ("human",
             """**User Query**: {user_query}
             """)
        ])
        
        research_chain = prompt | self.llm.with_structured_output(Output)
        response = research_chain.invoke({"user_query":user_query})
        #print("Queries:", response)
        return response
    
if __name__ == "__main__":
    user_query = "can you help write a topic about transformers?"
    r = QuerySuggester()
    response = r.research(user_query)
    print(response)
    for i in response.search_queries:
        print(i)
    