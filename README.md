# PHOTON - A Personal Desktop Assistant

**PHOTON (Personal Hyper-Optimized Tactical Operations Network)** is an advanced, voice-controlled desktop assistant inspired by JARVIS from Iron Man. Built with Python and powered by the Groq API with the Llama 3 model, PHOTON is designed to be a seamless and intelligent partner for your daily tasks. It can understand complex commands, perform actions on your computer, and even learn from documents you provide.

## Features

*   **Voice-Controlled Interaction:** Speak to PHOTON in natural language to get things done.
*   **Conversational AI:** Powered by the Llama 3 model via the Groq API for fast and intelligent responses.
*   **Task Automation:** PHOTON can perform a variety of tasks, including:
    *   Getting the current time and weather.
    *   Playing YouTube videos.
    *   Searching Wikipedia.
    *   Sending WhatsApp messages.
*   **Knowledge Base:** You can provide documents (PDFs, text files, etc.) to PHOTON, and it will learn from them to answer your questions.
*   **Agentic Capabilities:** PHOTON can delegate complex tasks to a specialized "agent" that can use a variety of tools to get the job done.
*   **Standalone Transcription Tool:** A separate tool is included to transcribe your speech into formatted text, either as an essay or a log entry.

## How It Works

PHOTON's workflow is a seamless integration of several key technologies:

1.  **Voice Input:** The assistant continuously listens for a hotword (holding the `space` key).
2.  **Speech-to-Text:** Your voice is recorded and transcribed into text using the `openai/whisper-medium` model.
3.  **Natural Language Understanding:** The transcribed text is sent to the Groq API, which uses the Llama 3 model to understand your intent.
4.  **Task Execution:**
    *   For simple conversational queries, PHOTON responds directly.
    *   For more complex tasks, PHOTON's "agent" is invoked. The agent uses a ReAct (Reasoning and Acting) framework to select the appropriate tool (e.g., weather tool, YouTube tool) to fulfill your request.
5.  **Text-to-Speech:** The assistant's response is converted back to a natural-sounding male voice using Microsoft Edge's TTS engine.

## Project Structure

The project is organized into the following key files and directories:

```
PHOTON---A-Personal-Desktop-Assistant/
├── photon/
│   ├── __init__.py
│   ├── agent.py
│   ├── photon.py
│   ├── tools.py
│   ├── transcriptions.py
│   ├── data/
│   └── message_history/
├── Pipfile
├── Pipfile.lock
└── README.md
```

*   **`photon/`**: The main package for the project.
    *   **`__init__.py`**: Makes the `photon` directory a Python package.
    *   **`photon.py`**: The core script that runs the main assistant.
    *   **`agent.py`**: The specialized agent that can use tools to perform tasks.
    *   **`tools.py`**: Contains the individual tools that the agent can use.
    *   **`transcriptions.py`**: The standalone transcription tool.
    *   **`data/`**: A directory where you can place your documents for PHOTON to learn from.
    *   **`message_history/`**: Stores the conversation history.
*   **`Pipfile` and `Pipfile.lock`**: These files manage the project's Python dependencies using `pipenv`.
*   **`README.md`**: This file.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PHOTON---A-Personal-Desktop-Assistant.git
    cd PHOTON---A-Personal-Desktop-Assistant
    ```

2.  **Install `pipenv`:**
    ```bash
    pip install pipenv
    ```

3.  **Install dependencies:**
    ```bash
    pipenv install
    ```

4.  **Set up API Keys:**
    *   This project requires API keys from Groq and OpenWeatherMap.
    *   You will need to replace the placeholder `"YOUR_API_KEY"` in the following files with your actual API keys:
        *   `photon/photon.py`
        *   `photon/agent.py`
        *   `photon/tools.py` (for OpenWeatherMap)

5.  **Activate the virtual environment:**
    ```bash
    pipenv shell
    ```

## Usage

1.  **Run the Assistant:**
    ```bash
    python photon/photon.py
    ```

2.  **Interact with PHOTON:**
    *   Press and hold the `space` bar to speak your command.
    *   To use the agent's tools, phrase your command like: "Photon, use your agent to..."

3.  **Use the Transcription Tool:**
    ```bash
    python photon/transcriptions.py
    ```
    *   Press and hold the `Delete` key to transcribe your speech into an essay format.
    *   Press and hold the `End` key to transcribe your speech into a log entry format.

## Available Tools

The agent has access to the following tools:

*   **`get_time`**: Gets the current time for a specific location.
*   **`weather_data`**: Fetches the current weather for a city.
*   **`youtube_transcript`**: Gets the transcript of a YouTube video.
*   **`send_whatsapp_message`**: Sends a WhatsApp message using the desktop application.
*   **`play_youtube_video`**: Plays a YouTube video in your browser.
*   **`wikipedia_search`**: Opens a Wikipedia search in your browser.
*   **`real_estate_database`**: Answers questions about real estate data from the `data/` directory.
*   **`retrieve_conversation_history`**: Summarizes your past conversation with PHOTON.

## Future Improvements

*   **Add more tools:** The agent's capabilities can be expanded by adding more tools for different tasks.
*   **Improve hotword detection:** Instead of holding a key, use a more sophisticated hotword detection engine.
*   **GUI:** Create a graphical user interface for the assistant.
*   **Secure API key management:** Use a more secure method for storing and accessing API keys, such as environment variables or a `.env` file.