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

                Generate exactly 3 diverse, formal academic search queries and exactly 1 concise reformulated research question based on the input.

                Rules:
                - Each search query must focus on a different conceptual angle.
                - If the input does not specify a domain, preserve breadth and avoid assuming one.
                - Use precise academic terminology.
                - Avoid generic or overly short phrases.
                - Do NOT include explanations, brackets, or meta commentary.
                
                The optimized_user_query must:
                - Be a full, formal academic research question or title.
                - Not be shorter than 6 words.
                - Not introduce a narrower scope than the original input.
                """),
            ("human",
             """**User Query**: {user_query}
             """)
        ])
        
        research_chain = prompt | self.llm.with_structured_output(QueryOutput)
        response = research_chain.invoke({"user_query":user_query})
        #print("Queries:", response)
        return response
    
if __name__ == "__main__":
    user_query = "can you wrtie me a topic about transformers"
    r = QuerySuggester()
    response = r.research(user_query)
    print(response)