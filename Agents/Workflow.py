from .QuerySuggester import QuerySuggester, QueryOutput
from .RelevanceChecker import RelevanceChecker
from .Researcher import Researcher, CitedOutput 
from .Verifier import Verifier, VerifierOutput 
from .Writer import Writer
from langgraph.graph import StateGraph, END
from langgraph.graph.state import Send
from typing import TypedDict, List, Annotated
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from RAG.retriever import ArxivRAGService
import operator


class AgentState(TypedDict):
    user_query: str
    optimized_user_query: str
    search_queries: QueryOutput
    documents: List[List[Document]]
    verified_drafts: Annotated[List[CitedOutput], operator.add] 
    final_paper: str
    retriever: EnsembleRetriever

class PaperState(TypedDict):
    user_query: str
    document: List[Document]
    is_relevant: bool 
    draft: CitedOutput
    verification_report: VerifierOutput
    
class AgentWorkflow:
    def __init__(self):
        self.query_suggester = QuerySuggester()
        self.writer = Writer()
        self.paper_workflow = PaperWorkflow()
        self.rag_service = ArxivRAGService()
        self.workflow = self.build_workflow()
        
    def build_workflow(self):
        agent_workflow = StateGraph(AgentState)
        
        agent_workflow.set_entry_point("suggest_query")
        
        agent_workflow.add_node("suggest_query", self.suggest_query)
        agent_workflow.add_node("retrieve_docs", self.retrieve_documents)

        # make the paper_research a pass-through node
        agent_workflow.add_node("paper_research", lambda state: state)

        agent_workflow.add_node("paper_workflow", self.paper_workflow.run)
        agent_workflow.add_node("write", self.write)

        agent_workflow.add_edge("suggest_query", "retrieve_docs")
        agent_workflow.add_edge("retrieve_docs", "paper_research")

        # This is the important part to run List[Send]
        agent_workflow.add_conditional_edges("paper_research", self.research)
        agent_workflow.add_edge("paper_workflow", "write")
        
        agent_workflow.set_finish_point("write")
        
        return agent_workflow.compile()
        
    def suggest_query(self, state: AgentState):
        query_output = self.query_suggester.research(state["user_query"]) 

        return {"search_queries":query_output.search_queries, 
                "optimized_user_query":query_output.optimized_user_query}
    
    def retrieve_documents(self, state: AgentState):
        retriever = self.rag_service.build_retriever(state["search_queries"])
        docs = retriever.invoke(state["optimized_user_query"])
        print("Docs length: ", len(docs))
        doc_dict = {}
        for doc in docs:
            doc_id = doc.metadata.get("entry_id")
            doc.metadata.pop("Summary")
            if doc_id not in doc_dict:
                doc_dict[doc_id] = []
                
            doc_dict[doc_id].append(doc)
    
        documents = list(doc_dict.values())
        print(len(documents))
        
        return {"retriever": retriever, "documents": documents}
    
    def research(self, state: AgentState):
        
        return [Send("paper_workflow", {"user_query": state["user_query"], "document": doc}) for doc in state["documents"]]

    def write(self, state: AgentState):
        synthesized_text = "\n"
        i = 1
        for v in state["verified_drafts"]:
            if isinstance(v, CitedOutput):
                synthesized_text += f"Draft {i}:\n{v.draft_text} \nREFERENCE: [{i}]{v.full_reference}\n\n"
                i+=1
                
        response = self.writer.write(state["user_query"], synthesized_text)
        return {"final_paper":response}
    
    def run(self, user_query: str):
        
        intial_state: AgentState ={
            "user_query":user_query,
            "optimized_user_query":None,
            "search_queries":None,
            "documents":None,
            "verified_drafts":[],
            "final_paper":None,
            "retriever":None
        }
        
        final_state = self.workflow.invoke(intial_state)
        return final_state
    

class PaperWorkflow:
    def __init__(self):
        self.relevance_checker = RelevanceChecker()
        self.researcher = Researcher()
        self.verifier = Verifier()
        self.workflow = self.build_workflow()
        
    def build_workflow(self):
        paper_graph = StateGraph(PaperState)
        paper_graph.add_node("check_relevance", self.check_relevance)
        paper_graph.add_node("research", self.research)
        paper_graph.add_node("verify", self.verify)
        
        paper_graph.set_entry_point("check_relevance")
        
        paper_graph.add_conditional_edges("check_relevance", self.is_relevance)
        paper_graph.add_edge("research","verify")
        paper_graph.add_conditional_edges("verify", self.is_verified)        
        
        return paper_graph.compile()
        
    def check_relevance(self, state: PaperState):
        top_k_chunks = [doc.page_content for doc in state["document"]]
        response = self.relevance_checker.check(state["user_query"],top_k_chunks)
        
        if response in ["CAN_ANSWER", "PARTIAL"]:
            return {"is_relevant": True,}
        elif response == "NO_MATCH":
            return {
                "is_relevant": False,
                "draft_response": "This question isn't related (or there's no data) for your query. Please ask another question relevant to the uploaded document(s)."
            }
    
    def is_relevance(self, state: PaperState):
        if state["is_relevant"]:
            return "research"
        return END
    
    def research(self, state: PaperState):
        if not state["document"]:
            raise ValueError("No documents found in state.")
        
        documents = [doc.page_content for doc in state["document"]]
        authors = state["document"][0].metadata["Authors"]
        published = state["document"][0].metadata["published_first_time"]
        title = state["document"][0].metadata["Title"]
        link = state["document"][0].metadata["links"][0]
        citation = f"{authors}, {title}, {published}, {link}"
        print("Metadata: ", citation)
        if not link or not authors or not title:
            return END    
        if not state["verification_report"]:
            response = self.researcher.research(state["user_query"],documents)
        else:   
            result = self.researcher.re_research(state["user_query"], documents, state["draft"].draft_text, state["verification_report"])
            response = state["draft"]
            response.draft_text = result
            
        draft = CitedOutput(draft_text=response, full_reference=citation)
        return {"draft": draft}
    
    def verify(self, state: PaperState):
        documents = [doc.page_content for doc in state["document"]]
        response = self.verifier.verify(state["user_query"], state["draft"].draft_text, documents)
        return {"verification_report": response}
    
    def is_verified(self, state: PaperState):
        if state["verification_report"].has_issues:
            if state["verification_report"].relevant:
                return END
            return "research"
        return END
    
    def run(self, state: PaperState):
        full_state: PaperState = {
            "user_query": state.get("user_query"),
            "document": state.get("document"),
            "is_relevant": False,
            "draft": "",
            "verification_report": None,
        }
        final_state = self.workflow.invoke(full_state)
        return {"verified_drafts": [final_state["draft"]]}