from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class Writer:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
    def write(self, user_query: str, verified_drafts: list) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """You are a professional academic paper writer. Follow these rules STRICTLY.

            **INPUT**:
            - You will receive multiple VERIFIED DRAFTS.
            - Each draft ends with a REFERENCE line in this exact format:
                REFERENCE: [n]Full reference
            where 'n' is a citation number.
            - Only use information explicitly present in these verified drafts.
            - Do NOT introduce any new information.

            **TASK**:
            1. Synthesize all drafts into ONE coherent academic paper.
            2. You MUST use the provided references when synthesizing the paper.
            3. Ensure logical flow, clarity, and a formal academic tone.
            4. Maintain factual consistency.
            5. Create a clear, descriptive title.

            **CITATION RULES**:
            - Every paragraph that contains information from a verified draft MUST end with at least one citation in square brackets.
            - Use ONLY the numbers provided in the REFERENCE lines of the verified drafts.
            - Reuse citation numbers if multiple sentences reference the same draft.
            - Do NOT invent new citations.

            **REFERENCE RULES**:
            - The reference list must use the exact citation numbers provided in the verified drafts.
            - Do NOT add references that are not present in the verified drafts.

            **OUTPUT FORMAT**:
            1. Title of the paper.
            2. Synthesized academic text with citations.
            3. Reference list corresponding to the citations used.
            4. Do NOT include anything else (no explanations, no extra text).
            """),
            ("human",
            """
            User Query:
            {user_query}

            Verified Drafts:
            {verified_drafts}

            Generate:
            1. A well-defined title.
            2. Final academic text with citations.
            3. Complete reference list.
            """)
        ])
        
        writer_chain = prompt | self.llm
        response = writer_chain.invoke({"user_query":user_query, "verified_drafts":verified_drafts})
        #print("writer: ", response)
        return response.content
