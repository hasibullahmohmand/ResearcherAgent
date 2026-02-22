from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class VerifierOutput(BaseModel):
    supported: bool = Field(description="True ONLY if every claim in the answer is explicitly supported by the context.")
    relevant: bool = Field(description="True if the answer directly addresses the user's specific question.")
    has_issues: bool = Field(description="True if the lists 'unsupported_claims' or 'contradictions' are not empty.")
    unsupported_claims: Optional[List[Dict[str, str]]] = Field(description="Key value pair of unsupported claims and full reasons of any unsupported claims if present.")
    contradictions: Optional[List[Dict[str,str]]] = Field(description="Key value pair of contradicitons and full reasons of any contradictions if present.")
    additional_details: Optional[List[str]] = Field(description="Any extra information or explanations")

class Verifier:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def verify(self, query:str, response:str, context:List[str]) -> VerifierOutput:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
            You are an expert RAG (Retrieval-Augmented Generation) Evaluator. Your task is to audit a candidate Response to a user Query based *solely* on the provided Context.

            **Guidelines for Evaluation:**

            1.  **Relevance Check:**
                - Does the Response directly address the user's Query?
                - If the response talks about the topic but ignores the specific constraint of the user query, mark `relevant` as False.

            2.  **Support & Hallucination Check:**
                - verify that every claim in the Response is supported by the Context.
                - **Do not** use your own outside knowledge. If a fact is true in the real world but missing from the Context, it is considered "Unsupported."
                - If the Response contains *any* information not found in the Context, `supported` must be False.

            """),
            ("human",
            """**User Query**: {query}
            
            **Response**: {response}
            
            **context**: {context}
            """)
        ])
  
        verify_chain = prompt | self.llm.with_structured_output(VerifierOutput)
        response = verify_chain.invoke({"query":query, "response":response, "context":context})
        #print(f"Verification response: {response}")
        return response

if __name__ == "__main__":
    v = Verifier()
    question = "What is Backpropagation?",
    answer = "Backpropagation is a supervised learning algorithm used to train neural networks.",
    context = "Backpropagation is a supervised learning algorithm used to train neural networks..."
    
    response = v.verify(question, answer, context)
    print(response)
        