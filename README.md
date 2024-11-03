# PHOTON - A Personal Desktop Assistant

PHOTON is an advanced personal desktop assistant inspired by JARVIS from Iron Man. Built using the Groq API and the Llama model, The assistant is equipped to handle complex command structures and can invoke agents and tools to execute specific tasks. It operates as a conversational AI with the capability to transcribe, summarize, and respond intelligently based on context.

## Project Structure

This project consists of several key files, each playing a crucial role in the assistant's functionality.

### 1. `photon.py`

The core of the PHOTON assistant, `photon.py` handles the main functionality, including:
- **Speech Recognition**: Converts voice input into text using a speech-to-text system.
- **Model Response Generation**: Sends the transcribed text to the Groq model, which processes the query and generates a response.
- **Text-to-Speech Conversion**: Converts the model's response back into speech for output.
- **Agent Invocation**: Based on the system prompt format, `photon.py` can call `agent.py` to handle specific tasks. If the model responds in a format that indicates an agent is needed, the message is routed to the agent, which then invokes necessary tools to fulfill the request.
  
The system prompt in `photon.py` defines the assistant's personality, behavior, and response style, ensuring PHOTON operates with precision and efficiency.

### 2. `agent.py`

`agent.py` is responsible for initializing and managing the agent. Key functions include:
- **Embedding Model Initialization**: Initializes an embedding model from HuggingFace for vector storage and retrieval.
- **Groq Model Initialization**: Sets up the Groq model to support agent operations.
- **Vector Storage**: Uses Llama Index's Simple Directory Reader to vectorize and store data for use in queries.
- **Agent Initialization**: The agent is set up to handle specific tasks and has access to various tools (imported from `tools.py`), which it can invoke based on user commands.
- **Storage Folder Creation**: When the agent processes and vectorizes data, it saves the index in a `storage/` folder for efficient retrieval in future interactions.

### 3. `tools.py`

`tools.py` houses various helper functions and tools required by the agent to perform specific actions. These tools are:
- **Specific Functions**: Each tool is a standalone function written to handle particular tasks requested by the agent.
- **Importable by Agent**: `tools.py` functions are imported into `agent.py`, making them accessible to the agent when specific operations are needed.
  
### 4. `transcriptions.py`

`transcriptions.py` operates independently from the main agent system, focusing solely on transcription tasks. Its functionalities include:
- **Speech-to-Text Conversion**: Similar to `photon.py`, this file uses a speech-to-text system to transcribe spoken input.
- **LLM Integration for Refinement**: Transcribed text is sent through the language model (LLM) for refinement and rephrasing, ensuring clear and coherent output.
- **Automated Typing**: Once refined, the output text is either displayed or typed out on the user’s computer.
  
### Additional Components

- **`data/` Folder**: Contains data files and any resources needed for your local knowledge base. It can include PDF's, TXT files, JSON files and more.
- **`storage/` Folder**: Automatically created by `agent.py` for storing vectorized data used in queries. This folder houses index files for efficient information retrieval.

## Setup and Installation

### Requirements

The project requires the following dependencies, which are listed in `requirements.txt`:

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Assistant**: Start the assistant by running `photon.py`:
   ```bash
   python photon.py
   ```

2. **Voice Interaction**: Speak to PHOTON to initiate commands. It will transcribe your speech, process it through the model, and respond with audio output.
3. **Agent Invocation**: If the assistant's response requires specific actions, it will automatically call `agent.py` to handle the request, using the tools from `tools.py` as needed.
4. **Transcriptions Only**: For transcription-focused tasks, use `transcriptions.py` independently to transcribe, refine, and output text without engaging the full conversational model.

## How It Works

1. **Speech to Text**: User speech is transcribed into text, which is then fed to the Groq model for processing.
2. **Model Processing**: The LLM in `photon.py` generates a response based on the prompt and context.
3. **Agent and Tools**: If the model determines an agent is needed, `agent.py` is called, which can use tools in `tools.py` to perform specific tasks.
4. **Text to Speech**: The assistant’s response is converted to speech and delivered to the user.

## Project Features

- **Conversational AI**: Engages users in natural language, handling tasks and queries seamlessly.
- **Contextual Transcription**: `transcriptions.py` provides transcription capabilities with context-aware rephrasing.
- **Tool-Based Functionality**: Tools in `tools.py` offer modular, expandable functionality for diverse operations.
- **Vectorized Query Storage**: Efficient data storage in `storage/` allows for quick retrieval in agent operations.

## Future Improvements

Potential areas for further development include:
- Expanding tool functions in `tools.py` for more complex tasks.
- Enhancing agent capabilities in `agent.py` for smarter decision-making.
- Adding more language support and refining transcription accuracy in `transcriptions.py`.
