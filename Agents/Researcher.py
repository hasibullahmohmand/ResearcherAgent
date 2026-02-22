from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
from Verifier import VerifierOutput

class CitedOutput(BaseModel):
    paragraph: str = Field(description="The drafted text based on the paper.")
    source_link: str = Field(description="The link provided in the prompt.")
    full_reference: str = Field(description="The properly formatted academic reference for this paper.")

class Researcher:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def research(self, user_query: str, context: List[str], citation: str, link: str) -> CitedOutput:

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are an expert academic researcher. Your task is to extract relevant information from the provided context to answer the user's query.

            STRICT RULES:
            1. CONTEXT ONLY: You must base your response *strictly* on the provided Context. Do not use outside knowledge.
            2. BE COMPREHENSIVE: Extract as much relevant detail as possible from the context that directly pertains to the user's query.
            """
             ),
            ("human",
            """**User Query**: {user_query}
            
            **Citation**: {citation}
            **Source Link**: {link}
            
            **Context**: 
            {context}
            """)
        ])
        
        research_chain = prompt | self.llm.with_structured_output(CitedOutput)
        
        response = research_chain.invoke({"user_query":user_query, "context":context, "citation":citation, "link":link})
        #print("Researcher Draft:", response)
        return response
    
    def re_research(self, user_query: str, context: List[str], drafted_response: CitedOutput, verification_report: VerifierOutput):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are an expert academic researcher. Your task is to CRITIQUE and REVISE a previously drafted response that failed a quality audit.
            
            STRICT RULES:
            1. REPAIR ONLY: Identify the 'unsupported' claims in the Drafted Response and replace them with verified facts from the Context.
            2. CONTEXT ONLY: If a claim in the draft isn't in the Context, delete it. Do not invent new facts.
            """),
            ("human",
            """
            **User Query**: {user_query}

            **Context**: 
            {context}

            ---
            **PREVIOUS DRAFTED RESPONSE (TO BE FIXED)**:
            "{drafted_response}"

            **VERIFICATION REPORT (ISSUES FOUND)**:
            - Relevant: {is_relevant}
            - Unsupported Claims: {unsupported}
            - Contradictions: {contradictions}
            - Additional Feedback: {feedback}
            
            **Instructions**: Provide a clean, corrected version of the response that addresses all issues in the report while maintaining the valid parts of the draft.
            """)
        ])
        
        unsupported_str = "\n".join([f"- {list(d.keys())[0]}: {list(d.values())[0]}" for d in (verification_report.unsupported_claims or [])])
        contradictions_str = "\n".join([f"- {list(d.keys())[0]}: {list(d.values())[0]}" for d in (verification_report.contradictions or [])])

        research_chain = prompt | self.llm.with_structured_output(CitedOutput)
        
        response = research_chain.invoke({
            "user_query": user_query, 
            "context": context, 
            "drafted_response": drafted_response,
            "is_relevant": verification_report.relevant,
            "unsupported": unsupported_str if unsupported_str else "None",
            "contradictions": contradictions_str if contradictions_str else "None",
            "feedback": ". ".join(verification_report.additional_details or ["No additional feedback"])
        })
        #print("Researcher Revised Draft:", response)
        return response
    
if __name__ == "__main__":
    query = "can you help write a topic about transformers?"
    context = """A Deep Learning Transformer is a neural network architecture that uses self-attention mechanisms to process
        sequential data, such as text or time series data. Introduced in 2017, Transformers have revolutionized the field
        of natural language processing (NLP) and have achieved state-of-the-art results on various tasks, including
        machine translation, text summarization, and question answering. By allowing the model to attend to all positions
        in the input sequence simultaneously, Transformers can capture long-range dependencies and relationships, making
        them particularly well-suited for tasks that require complex contextual understanding."""
    citation = "Attention Is All You Need, 2018, Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin"
    link = "https://arxiv.org/abs/1706.03762v7"
    r = Researcher()
    response = r.research(query, [context], citation, link)
    print(response)