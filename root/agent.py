import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq  # Import Groq API LLM
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from tools import time_engine, weather_tool, yt_transcript_tool, send_whatsapp_message_tool, play_youtube_tool, open_wikipedia_search_tool
from llama_index.core.tools import FunctionTool
from llama_index.core.composability import QASummaryQueryEngineBuilder

# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize the HuggingFace embedding model (replace with your desired model)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Custom HuggingFace model

# Initialize the Groq API LLM
llm = Groq(model="llama3-70b-8192", api_key="YOUR_API_KEY")
Settings.llm = llm  # Set the LLM in the Settings object to use the Groq API LLM

# Load documents from the 'data' directory
real_estate_bangalore = SimpleDirectoryReader("data").load_data()
conversation_history = SimpleDirectoryReader("message_history").load_data()

# Create a VectorStoreIndex from the documents using the custom HuggingFace embedding model
index_real_estate = VectorStoreIndex.from_documents(real_estate_bangalore, embed_model=embed_model)
index_real_estate.storage_context.persist(persist_dir="./storage/real_estate_bangalore")

engine_real_estate = index_real_estate.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

query_engine_builder = QASummaryQueryEngineBuilder(embed_model=embed_model)
conversation_history_engine = query_engine_builder.build_from_documents(conversation_history)

def real_estate_chat_engine(input_text: str) -> str:
    """
    Wrapper function that takes user input, feeds it to engine.chat(), 
    and returns the response.
    
    :param input_text: The input text to be passed to the engine.
    :return: The response from engine.chat().
    """
    response = engine_real_estate.chat(input_text)
    return response

real_estate_engine = FunctionTool.from_defaults(real_estate_chat_engine, name="real_estate_database")

def retrieve_conversation_history() -> str:
    """
    Retrieve and summarize the conversation history from a JSON file.
    
    :return: A detailed summary of the conversation history with more details for recent conversations.
    """
    try:
        response = conversation_history_engine.query(
            "Provide a summary of the conversation history in detail. Provide more details for recent conversations compared to older ones."
        )
        
        return response
    
    except Exception as e:
        return f"Error retrieving conversation history: {e}"

# Wrap the function with FunctionTool
retrieve_conversation_history_tool = FunctionTool.from_defaults(
    retrieve_conversation_history,
    name="retrieve_conversation_history",
    description="Retrieve and summarize the conversation history from the JSON file."
)

# Define the query engine tools
query_engine_tools = [
    time_engine,
    real_estate_engine,
    weather_tool,
    yt_transcript_tool,
    retrieve_conversation_history_tool,
    send_whatsapp_message_tool,
    play_youtube_tool,
    open_wikipedia_search_tool
]

# Initialize the ReActAgent with the tools
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, max_iterations=20, verbose=True, memory=None, context="You MUST respond in the lowest iterations possible. If you can't find the answer in the tool, you must say so. DO NOT use unnecessary tools.")
agent.reset()
