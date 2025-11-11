import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

def create_agent(tools):
    """
    Creates a RAG agent that can access documents and use tools.
    """
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Initialize the HuggingFace embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the Groq API LLM
    llm = Groq(model="llama-3.3-70b-versatile", api_key="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")
    Settings.llm = llm

    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader("data").load_data()

    # Create a VectorStoreIndex from the documents
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Create a query engine tool
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="rag_tool",
        description="This tool can answer questions about the documents in the data directory.",
    )

    # Add the query engine tool to the list of tools
    all_tools = tools + [query_engine_tool]

    # Initialize the ReActAgent with the tools
    agent = ReActAgent.from_tools(all_tools, llm=llm, max_iterations=20, verbose=True, memory=None, context="You MUST respond in the lowest iterations possible. If you can't find the answer in the tool, you must say so. DO NOT use unnecessary tools.")
    agent.reset()
    return agent
