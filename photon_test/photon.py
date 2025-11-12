import os
import asyncio
import tempfile
import json
import numpy as np
import sounddevice as sd
import wavio
import keyboard  # To detect key press
import soundfile as sf
from threading import Thread
from queue import Queue
from collections import deque
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
import edge_tts

# ANSI color codes
class colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Initialize Groq client
groq_client = Groq(api_key="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")

from tools import (
    search_wikipedia_and_open,
    get_wikipedia_article_summary,
    play_youtube_video_by_title,
    get_weather_forecast,
    get_current_datetime,
    document_qa_tool,
    write_to_notepad,
)

# Agent creation
def create_agent():
    """
    Creates a RAG agent that can access documents and use tools.
    """
    # Initialize the Groq API LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")

    # Define the tools for the agent
    tools = [
        search_wikipedia_and_open,
        play_youtube_video_by_title,
        get_weather_forecast,
        get_current_datetime,
        document_qa_tool,
        get_wikipedia_article_summary,
        write_to_notepad,
    ]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are PHOTON - a Personal Hyper-Optimized Tactical Operations Network. Your persona is inspired by Jarvis from Iron Man: you are sophisticated, witty, and incredibly helpful. Your primary function is to assist the user with a wide range of tasks by leveraging the tools at your disposal. Always be proactive and anticipate the user's needs. When a query is simple and does not require a tool, provide a direct and concise answer. When a tool is necessary, use it efficiently and intelligently. If a tool fails, inform the user gracefully and suggest an alternative course of action. Your responses should be clear, articulate, and reflect your advanced capabilities. Above all, maintain a professional and courteous demeanor, Sir."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

# Create the agent
print(f"{colors.BOLD}{colors.GREEN}Initializing PHOTON...{colors.ENDC}")
agent_executor = create_agent()
print(f"{colors.BOLD}{colors.GREEN}PHOTON is online and ready.{colors.ENDC}")

# Conversation history and TTS audio queue
conversation_history = deque(maxlen=6)
audio_queue = Queue()
history_file_path = "message_history/conversation_history.json"
os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

# Audio recording parameters
sample_rate = 16000
audio_filename = "temp_audio.wav"
chunk_duration = 1  # Record in chunks of 1 second

# Function to save conversation history to JSON
def save_conversation_history(entry, file_path):
    try:
        with open(file_path, "a") as file:
            json.dump(entry, file)
            file.write("\n")
    except Exception as e:
        print(f"{colors.RED}Error saving conversation history: {e}{colors.ENDC}")

# Function to synthesize text to audio and add to queue
async def synthesize_text_to_audio(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpfile:
        filename = tmpfile.name
    communicate = edge_tts.Communicate(text, "en-CA-LiamNeural", pitch="-40Hz")
    await communicate.save(filename)
    return filename

async def synthesize_and_enqueue_audio(sentence):
    print(f"{colors.YELLOW}Synthesizing audio for response...{colors.ENDC}")
    audio_file = await synthesize_text_to_audio(sentence)
    audio_queue.put(audio_file)
    print(f"{colors.GREEN}Audio synthesis complete.{colors.ENDC}")

def process_audio_queue():
    while True:
        audio_file = audio_queue.get()
        try:
            print(f"{colors.YELLOW}Playing audio response...{colors.ENDC}")
            data, fs = sf.read(audio_file, dtype='float32')
            sd.play(data, fs)
            sd.wait()
            print(f"{colors.GREEN}Audio playback finished.{colors.ENDC}")
        except Exception as e:
            print(f"{colors.RED}Error playing sound: {e}{colors.ENDC}")
        finally:
            os.remove(audio_file)
        audio_queue.task_done()

def start_audio_queue_processor():
    audio_thread = Thread(target=process_audio_queue, daemon=True)
    audio_thread.start()

# Start TTS audio processor
start_audio_queue_processor()

def record_audio_continuously(sample_rate):
    """
    Records audio until the key is released.
    """
    print(f"\n{colors.BOLD}Listening... Press and hold 'space' to speak.{colors.ENDC}")
    audio_data = []

    while keyboard.is_pressed('space'):
        chunk = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_data.append(chunk)

    print(f"{colors.YELLOW}Recording finished.{colors.ENDC}")
    recording = np.concatenate(audio_data, axis=0)
    wavio.write(audio_filename, recording, sample_rate, sampwidth=2)
    return audio_filename

while True:
    print(f"\n{colors.BOLD}Waiting for user input... (Press and hold 'space' to speak or type 'exit' to stop){colors.ENDC}")
    keyboard.wait('space')

    # Record voice input when space is held down
    audio_file = record_audio_continuously(sample_rate)

    try:
        # Transcribe the audio file with Whisper
        print(f"{colors.YELLOW}Transcribing audio...{colors.ENDC}")
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3-turbo",
                temperature=0,
                response_format="verbose_json",
            )
        user_input = transcription.text
        print(f"{colors.GREEN}Transcription complete.{colors.ENDC}")
        print(f"{colors.BOLD}{colors.BLUE}User: {user_input}{colors.ENDC}")

        # Add to conversation history
        conversation_history.append({'role': 'user', 'content': user_input})
        save_conversation_history({'role': 'user', 'content': user_input}, history_file_path)

        # Get response from the agent
        print(f"\n{colors.YELLOW}PHOTON is thinking...{colors.ENDC}")
        agent_response = agent_executor.invoke({"input": user_input, "chat_history": list(conversation_history)})
        ai_response = agent_response['output']
        print(f"{colors.BOLD}{colors.GREEN}PHOTON: {ai_response}{colors.ENDC}")

        conversation_history.append({'role': 'assistant', 'content': ai_response})
        save_conversation_history({'role': 'assistant', 'content': ai_response}, history_file_path)

        # Synthesize and play AI response
        if ai_response:
            asyncio.run(synthesize_and_enqueue_audio(ai_response))

    except Exception as e:
        print(f"{colors.RED}An error occurred: {e}{colors.ENDC}")
        print(f"{colors.YELLOW}The audio file has been saved for later use.{colors.ENDC}")

