import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate

def create_agent(tools):
    """
    Creates a RAG agent that can access documents and use tools.
    """
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Initialize the HuggingFace embedding model
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the Groq API LLM
    llm = ChatGroq(model="llama3-70b-8192", api_key="")

    # Load documents from the 'data' directory
    loader = DirectoryLoader("data", glob="**/*.txt")
    documents = loader.load()

    # Create a VectorStoreIndex from the documents
    vector_store = FAISS.from_documents(documents, embed_model)
    retriever = vector_store.as_retriever()

    # Create a retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "rag_tool",
        "This tool can answer questions about the documents in the data directory.",
    )

    # Add the retriever tool to the list of tools
    all_tools = tools + [retriever_tool]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. You may not need to use tools for every query. Respond directly if the query does not require a tool."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, all_tools, prompt)

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

    return agent_executor