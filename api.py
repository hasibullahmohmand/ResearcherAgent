from fastapi import FastAPI
from Agents import AgentWorkflow
from pydantic import BaseModel
import uvicorn

class ApiOutput(BaseModel):
    final_paper: str
    researched_drafts: str
    paper_research_terms: str
    

app = FastAPI()
workflow = AgentWorkflow()

@app.post("/agent")
def process_user_query(user_query: str) -> ApiOutput:
    if not user_query.strip():
        raise ValueError("Please enter a research query.")

    response = workflow.run(user_query)

    paper = response.get("final_paper", "")
    drafts = response.get("verified_drafts", "")
    paper_research = response.get("search_queries", "")
    
    synthesized_text = "\n"
    i = 1
    for v in drafts:
        if not isinstance(v, str):
            synthesized_text += f"Draft {i}:\n {v.draft_text} \n REFERENCE: [{i}]{v.full_reference}\n\n"
            i+=1
    if isinstance(paper_research, list):
        paper_research = "\n".join(paper_research)

    return ApiOutput(
        final_paper=paper,
        researched_drafts=synthesized_text,
        paper_research_terms=paper_research
    )
    
if __name__ == "__main__":
    uvicorn.run(app)