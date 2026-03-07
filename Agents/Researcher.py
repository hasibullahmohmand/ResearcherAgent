from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
    
class CitedOutput(BaseModel):
    draft_text: str = Field(description="The full drafted text based on the paper.")
    full_reference: str = Field(description="The properly formatted academic reference for this paper in IEEE style with Author(s), Title, Year, and source link.")

class Researcher:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def research(self, user_query: str, context: List[str]) -> str:
        
        context_text = "\n".join(context)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are an expert academic researcher.

            Your task is to write a detailed academic draft that answers the user's query using ONLY the provided Context.

            **RULES:**
            1. Extract as much relevant information as possible from the Context.
            2. Write detailed explanations in continuous academic prose.
            3. DO NOT SUMMARIZE briefly.
            4. Use ONLY the provided Context. Do not add external knowledge.
            5. Remove citation markers like [1], [2] or other noisy references.
            6. Do NOT use bullet points, numbered lists, headings, or commentary.
            7. Output ONLY the draft text."""),
            ("human",
            """
            **User Query:**
            {user_query}

            **Context:**
            {context}

            Write a detailed draft answering the query using only the Context above.
            """)
            ])
                
        research_chain = prompt | self.llm
        
        response = research_chain.invoke({"user_query":user_query, "context":context_text})
        print("Researcher Draft:", response.content)
        return response.content
    
    def re_research(self, user_query: str, context: List[str], drafted_response: str, verification_report: BaseModel):
        
        context_text = "\n\n".join(context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are an expert academic researcher. Your task is to CRITIQUE and REVISE a previously drafted response that failed a quality audit.
            
            **STRICT RULES:**
            1. REPAIR ONLY: Identify the 'unsupported' claims in the Drafted Response and replace them with specific, verified facts from the Context.
            2. CONTEXT ONLY: If a claim in the draft isn't supported in the Context, delete it or replace it with what IS in the Context.
            3. BE SPECIFIC: Generate detailed content with concrete references from the Context.
            """),
            ("human",
            """P
            **User Query**: {user_query}

            **Paper Context**: 
            {context}

            ---
            **PREVIOUS DRAFTED RESPONSE (TO BE FIXED)**:
            "{drafted_response}"

            **ISSUES FOUND TO FIX**:
            - Relevant: {is_relevant}
            - Unsupported Claims: {unsupported}
            - Contradictions: {contradictions}
            - Feedback: {feedback}
            
            **Task**: Provide a revised paragraph that fixes these issues while maintaining accuracy to the paper content.
            """)
        ])
        
        unsupported_str = "\n".join([f"- {list(d.keys())[0]}: {list(d.values())[0]}" for d in (verification_report.unsupported_claims or [])])
        contradictions_str = "\n".join([f"- {list(d.keys())[0]}: {list(d.values())[0]}" for d in (verification_report.contradictions or [])])

        research_chain = prompt | self.llm
        
        response = research_chain.invoke({
            "user_query": user_query, 
            "context": context_text, 
            "drafted_response": drafted_response,
            "is_relevant": verification_report.relevant,
            "unsupported": unsupported_str if unsupported_str else "None",
            "contradictions": contradictions_str if contradictions_str else "None",
            "feedback": ". ".join(verification_report.additional_details or ["No additional feedback"])
        })
        print("Researcher Revised Draft:", response.content)
        
        return response.content
    
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
    response = r.research(query, [context])
    print(response)