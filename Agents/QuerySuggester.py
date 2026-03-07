from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Annotated

class QueryOutput(BaseModel):
    search_queries: Annotated[List[str],Field(min_length=3, max_length=3, description="3 search queries for finding relevant academic papers.")] 
    optimized_user_query: str = Field(description="A reformulated, keyword-dense version of the original user query optimized for academic retrieval.")

class QuerySuggester:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0.2)
    def research(self, user_query: str) -> QueryOutput:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an academic researcher specialized in information retrieval.

                Generate exactly 3 diverse, formal academic search queries and exactly 1 concise reformulated research query based on the user query.

                Rules:
                - Each search query must focus on a different conceptual angle.
                - Avoid generic or overly short phrases.
                - Do NOT include explanations, brackets, or meta commentary.
                """),
            ("human",
             """**User Query**: {user_query}
             """)
        ])
        
        research_chain = prompt | self.llm.with_structured_output(QueryOutput)
        response = research_chain.invoke({"user_query":user_query})
        print("Queries:", response)
        return response
    
if __name__ == "__main__":
    user_query = "can you write a topic about the impact of transformer models in image classification?"
    r = QuerySuggester()
    response = r.research(user_query)
    #print(response)