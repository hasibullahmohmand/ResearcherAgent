from Agents import AgentWorkflow
import gradio as gr

workflow = AgentWorkflow()


def process_user_query(user_query):
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
        paper_research = "".join(paper_research)

    return paper, synthesized_text, paper_research


css = """
.markdown-box {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 8px;
    background-color: #1e1e1e;
    color: #f5f5f5;
    height: 400px;
    overflow-y: auto;
}
.paper-research-box {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 8px;
    background-color: #1e1e1e;
    color: #f5f5f5;
    height: 100px;
}

"""

with gr.Blocks(title="Researcher Agent") as demo:

    gr.Markdown("# 📚 Researcher Agent")
    gr.Markdown(
        "Your AI research assistant that searches **arXiv papers** and writes academic summaries."
    )

    with gr.Row():

        # LEFT SIDE
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Generated Paper")

            paper_box = gr.Markdown(
                value="""### Your generated research paper will appear here

                Once you submit a query, the Researcher Agent will:

                - 🔍 Search relevant **academic papers**
                - 📑 Extract and verify information
                - ✍️ Generate a **structured research summary with citations**

                Try asking something like:

                **"Applications of deep learning in medical imaging"**""",
                elem_classes="markdown-box"
            )

            user_query = gr.Textbox(
                placeholder="Example: Deep learning for breast cancer classification",
                lines=2,
                label="Research Query"
            )

            submit_btn = gr.Button("Generate Paper", variant="primary")

        # RIGHT SIDE
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Research Drafts")

            drafts_box = gr.Markdown(
                value="""### Research Drafts Will Appear Here

                This panel shows **intermediate drafts** extracted from research papers before the final synthesis.

                These drafts help the agent:

                - 📚 gather information from multiple sources  
                - 🔎 verify factual accuracy  
                - 🧠 synthesize a coherent research summary  

                After you submit a query, the drafts collected from the papers will appear here.
                """,
                elem_classes="markdown-box"
            )
            gr.Markdown("### 🔎 Search Terms")
            paper_research_box = gr.Markdown(
                value="""### Research Paper Terms Will Appear Here
                This panel shows **Search Terms** that the Agent used to retrieve papers.
                """,
                elem_classes="paper-research-box"
            )

            with gr.Accordion("ℹ️ How it works", open=False):
                gr.Markdown(
                    """
                    1. Enter a research query.
                    2. The agent searches relevant papers from **arXiv**.
                    3. Important information is extracted and verified.
                    4. A structured research summary with references is generated.
                    """
                )

    submit_btn.click(
        process_user_query,
        inputs=user_query,
        outputs=[paper_box, drafts_box, paper_research_box],
    )

demo.launch(server_name="127.0.0.1", server_port=5000, css=css)