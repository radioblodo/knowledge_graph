#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Streamlit interface
import streamlit as st

# For PDF Text Extraction
import fitz

# For Knowledge Extraction (LangChain)
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from the .env file 
load_dotenv() 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Graph Visualization 
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Knowledge Graph Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Controls")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
user_focus = st.sidebar.text_input(
    "Optional: Highlight topics, entities, or questions to focus on.",
    placeholder="e.g., machine learning concepts",
)
max_nodes = st.sidebar.slider(
    "Max nodes to visualize",
    min_value=5,
    max_value=60,
    value=20,
    help="Limit the number of nodes that are rendered in the graph.",
)
generate = st.sidebar.button("Generate graph", use_container_width=True)

if not os.getenv("OPENAI_API_KEY"):
    st.sidebar.warning("OPENAI_API_KEY is not set. Please configure it in your environment.")

theme_label_map = {
    ":material/light_mode:": "Light",
    ":material/dark_mode:": "Dark",
}
spacer_col, theme_col = st.columns([0.8, 0.2])
with theme_col:
    selected_theme = st.radio(
        "Theme toggle",
        list(theme_label_map.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="theme_toggle",
    )
theme_choice = theme_label_map[selected_theme]

theme_tokens = {
    "Light": {
        "bg": "#f8fafc",
        "card": "#ffffff",
        "card_border": "rgba(148,163,184,0.4)",
        "title": "#0f172a",
        "subtitle": "#475569",
        "node": "#6366f1",
        "edge": "#cbd5f5",
        "graph_bg": "#ffffff",
        "graph_font": "#1f2937",
    },
    "Dark": {
        "bg": "#0f172a",
        "card": "#111826",
        "card_border": "rgba(148,163,184,0.25)",
        "title": "#e2e8f0",
        "subtitle": "#cbd5f5",
        "node": "#a78bfa",
        "edge": "#a3bffa",
        "graph_bg": "#0f172a",
        "graph_font": "#f1f5f9",
    },
}
active_theme = theme_tokens[theme_choice]

st.markdown(
    f"""
    <style>
    body {{
      font-family: "Inter", "Segoe UI", system-ui, -apple-system;
      background-color: {active_theme["bg"]};
    }}
    .stApp {{
      background-color: {active_theme["bg"]};
    }}
    .block-container {{
      padding-top: 1.4rem;
    }}
    .kg-card {{
      background: {active_theme["card"]};
      border-radius: 18px;
      padding: 32px;
      box-shadow: 0 30px 80px rgba(15, 23, 42, 0.20);
      margin-bottom: 26px;
      border: 1px solid {active_theme["card_border"]};
    }}
    .kg-title {{
      font-size: 34px;
      font-weight: 800;
      margin-bottom: 8px;
      color: {active_theme["title"]};
    }}
    .kg-subtitle {{
      color: {active_theme["subtitle"]};
      font-size: 18px;
      margin-bottom: 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="kg-card">
      <div class="kg-title">🧠 Knowledge Graph Builder</div>
      <p class="kg-subtitle">
        Use the sidebar on the left to upload a PDF, adjust the node limit, choose a theme,
        and optionally guide the extraction focus. The graph renders to the right.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


class Node(BaseModel):
    id: str = Field(description="Unique identifier for a node.")
    label: str = Field(description="The display name of the node.")
    type: str = Field(description="The category or type of the entity.")


class Edge(BaseModel):
    source: str = Field(description="The ID of the source node.")
    target: str = Field(description="The ID of the target node.")
    label: str = Field(
        description="The description of the relationship between nodes."
    )


class KnowledgeGraph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


def build_graph_from_pdf(pdf_bytes: bytes, focus: str) -> KnowledgeGraph:
    """Extract text from PDF bytes and run the LLM graph extraction chain."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert at extracting information and creating knowledge graphs.
        Based on the following text, extract the key entities (nodes) and their
        relationships (edges). Refine your extraction based on the user's focus if one
        is provided.

        User's focus: {user_focus}

        {format_instructions}

        Text to analyse:
        {text}
        """
    )

    chain = prompt_template | llm | parser

    return chain.invoke(
        {
            "text": full_text,
            "user_focus": focus or "general topics",
            "format_instructions": parser.get_format_instructions(),
        }
    )


def limit_graph(graph: KnowledgeGraph, limit: int) -> KnowledgeGraph:
    """Trim the output graph to the desired node count and related edges."""
    trimmed_nodes = graph.nodes[:limit]
    allowed_ids = {node.id for node in trimmed_nodes}
    trimmed_edges = [
        edge
        for edge in graph.edges
        if edge.source in allowed_ids and edge.target in allowed_ids
    ]
    return KnowledgeGraph(nodes=trimmed_nodes, edges=trimmed_edges)


if generate:
    if uploaded_pdf is None:
        st.warning("Please upload a PDF to build a graph.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Cannot continue without OPENAI_API_KEY configured.")
    else:
        with st.spinner("Extracting insights and rendering your knowledge graph..."):
            try:
                graph = build_graph_from_pdf(uploaded_pdf.getvalue(), user_focus)
            except Exception as exc:
                st.error(f"Unable to build graph: {exc}")
            else:
                if not graph or not graph.nodes:
                    st.info("No graph data returned by the LLM. Try refining the focus.")
                    st.json(graph.dict() if graph else {})
                else:
                    graph = limit_graph(graph, max_nodes)
                    net = Network(
                        directed=True,
                        height="750px",
                        width="100%",
                        bgcolor=active_theme["graph_bg"],
                        font_color=active_theme["graph_font"],
                    )
                    for node in graph.nodes:
                        net.add_node(
                            node.id,
                            label=node.label,
                            title=f"Type: {node.type}",
                            color=active_theme["node"],
                        )

                    for edge in graph.edges:
                        net.add_edge(
                            edge.source,
                            edge.target,
                            label=edge.label,
                            color=active_theme["edge"],
                        )

                    net.set_options(
                        """
                        var options = {
                          "interaction": {
                            "hover": true,
                            "navigationButtons": true,
                            "keyboard": { "enabled": true },
                            "zoomView": true,
                            "tooltipDelay": 120
                          },
                          "physics": {
                            "stabilization": { "iterations": 250 },
                            "solver": "forceAtlas2Based",
                            "forceAtlas2Based": { "springLength": 160 }
                          }
                        };
                        """
                    )
                    graph_html = net.generate_html()
                    components.html(graph_html, height=800, scrolling=False)
