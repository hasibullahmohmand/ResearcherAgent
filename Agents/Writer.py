from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class WriterOutput(BaseModel):
    final_text: str = Field(description="The final synthesized academic text based on the verified drafts.")
    full_reference: str = Field(description="The properly formatted academic reference including the link.")

class Writer:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def write(self, user_query: str, verified_drafts: list) -> WriterOutput:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
            You are a professional academic paper writer.

            You will receive multiple VERIFIED DRAFTS.
            Each draft already contains citation metadata and source links.

            Your task:

            1. Synthesize all verified drafts into ONE coherent academic paragraph.
            2. Ensure logical flow and formal academic tone.
            3. Do NOT introduce any new information.
            4. Use ONLY information explicitly present in the verified drafts.
            5. Remove redundancy.
            6. Maintain factual consistency.

            CITATION RULES:
            - Every sentence MUST end with a numbered citation in square brackets.
            - Use incremental numbering: [1], [2], [3], etc.
            - If multiple sentences use the same source, reuse the same number.

            REFERENCE RULES:
            - After the paragraph, provide a "References" section.
            - Format references in IEEE style.
            - Include: Author(s), Title, Year, and the provided URL.
            """),
            ("human",
            """
            User Query:
            {user_query}

            Verified Drafts:
            {verified_drafts}

            Generate:
            1. Final academic text
            2. Full academic reference
            """)
        ])
        
        writer_chain = prompt | self.llm.with_structured_output(WriterOutput)
        response = writer_chain.invoke({"user_query":user_query, "verified_drafts":verified_drafts})
        
        return response
