import os
import asyncio
import tempfile
import json
import re
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
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import edge_tts
import requests
import time
import pyautogui
import webbrowser
from datetime import datetime
from pytz import timezone, all_timezones
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from email.mime.text import MIMEText
import smtplib
from youtube_transcript_api import YouTubeTranscriptApi
import pywhatkit
from fuzzywuzzy import fuzz, process

# Initialize Groq client
groq_client = Groq(api_key="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")

# Tool definitions

@tool
def open_wikipedia_search(query: str) -> str:
    """
    Opens Wikipedia in the default web browser and searches for the provided query.

    :param query: The search term to look up on Wikipedia.
    :return: Confirmation message of action taken.
    """
    url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    webbrowser.open(url)
    return f"Wikipedia page opened for query: '{query}'"


@tool
def play_youtube_video(video_title: str) -> str:
    """
    Searches for a YouTube video by title and plays it in the browser.

    :param video_title: The title or keywords of the YouTube video to search and play.
                        This can be a partial or full title.
    :return: A confirmation message indicating the video is playing.
    """
    try:
        # Use pywhatkit to search for and play the video
        pywhatkit.playonyt(video_title)
        return f"Playing video: {video_title}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Dictionary of contacts with names as keys and phone numbers as values
contacts = {
    #Add your contacts here
}

@tool
def send_whatsapp_message(contact_name: str, message: str) -> str:
    """
    Sends a WhatsApp message to a specified contact using a desktop application. 
    If the contact name is not found, it will attempt to match using fuzzy search.

    :param contact_name: The name of the contact to send a message to.
    :param message: The message content to be sent to the contact.
    :return: A status message indicating the success or failure of the operation.
    """
    # Find the closest matching contact using fuzzy search
    match, score = process.extractOne(contact_name, contacts.keys(), scorer=fuzz.ratio)
    
    if score < 70:
        return "No close match found for the contact name."

    # Retrieve the phone number for the matched contact name
    phone_number = contacts[match]
    
    # Open WhatsApp Desktop
    pyautogui.hotkey('win', 's')
    time.sleep(1)
    pyautogui.write('WhatsApp')
    pyautogui.press('enter')
    time.sleep(5)  # Adjust based on loading time of WhatsApp

    # Search for the contact and select it
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(1)
    pyautogui.write(phone_number)
    time.sleep(1)
    pyautogui.press('down')
    pyautogui.press('enter')

    # Type and send the message
    time.sleep(1)
    pyautogui.write(message)
    pyautogui.press('enter')
    return f"Message sent to {match}: {message}"


@tool
def get_video_transcript(video_id: str) -> str:
    """
    Fetches the transcript for a YouTube video by ID and writes it to a file.

    :param video_id: The YouTube video ID (can start with a dash).
    :param languages: List of preferred languages for the transcript (default is English and German).
    :return: A success message or an error message if something went wrong.
    """
    # Handle video ID that may start with a dash
    video_id = video_id.lstrip('-')
    
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Prepare transcript text
        transcript_text = "\n".join(entry['text'] for entry in transcript)
        
        # Write transcript to file
        with open("transcript_output.txt", "a") as file:
            file.write(transcript_text + "\n")
        
        return f"Transcript for video '{video_id}' has been saved to 'transcript_output.txt'."
    
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def get_current_weather(city: str) -> dict:
    """
    Fetches current weather data for a given city from OpenWeather API.

    :param city: The name of the city to get weather data for.
    :return: A dictionary containing weather information, or an error message.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": "API_KEY",
        "units": "metric"  # Use 'imperial' for Fahrenheit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Extract relevant information
        weather = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "weather": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        
        return weather
    
    except requests.RequestException as e:
        return {"error": str(e)}


@tool
def get_current_time(location: str = "UTC") -> str:
    """
    Get the current time in a more human-readable format for a specific location.
    :param location: Timezone location for which to get the time. Defaults to "UTC".
    :return: The current time at the specified location.
    """
    if location not in all_timezones:
        return f"Error: '{location}' is not a recognized timezone."

    # Set timezone based on location
    tz = timezone(location)
    now = datetime.now(tz)
    current_time = now.strftime("%I:%M:%S %p")  # 12-hour format with AM/PM
    current_date = now.strftime("%A, %B %d, %Y")  # Full weekday, month name, day, and year

    return f"Current Date and Time in {location} = {current_date}, {current_time}"

# Agent creation
def create_agent():
    """
    Creates a RAG agent that can access documents and use tools.
    """
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Initialize the HuggingFace embedding model
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the Groq API LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")

    # Load documents from the 'data' directory
    loader = DirectoryLoader(
    'data/',
    glob="**/*.txt",        # Only load .txt files
    loader_cls=TextLoader   # Use TextLoader instead of Unstructured
)
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

    # Define the tools for the agent
    tools = [
        open_wikipedia_search,
        play_youtube_video,
        send_whatsapp_message,
        get_video_transcript,
        get_current_weather,
        get_current_time,
        retriever_tool,
    ]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are PHOTON - Personal Hyper-Optimized Tactical Operations Network, an advanced assistant with a demeanor and personality inspired by Jarvis from Iron Man. You are a helpful assistant. You may not need to use tools for every query. Respond directly if the query does not require a tool."),
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
agent_executor = create_agent()

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
        print(f"Error saving conversation history: {e}")

# Function to synthesize text to audio and add to queue
async def synthesize_text_to_audio(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpfile:
        filename = tmpfile.name
    communicate = edge_tts.Communicate(text, "en-CA-LiamNeural", pitch="-40Hz")
    await communicate.save(filename)
    return filename

async def synthesize_and_enqueue_audio(sentence):
    audio_file = await synthesize_text_to_audio(sentence)
    audio_queue.put(audio_file)

def process_audio_queue():
    while True:
        audio_file = audio_queue.get()
        try:
            data, fs = sf.read(audio_file, dtype='float32')
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"Error playing sound: {e}")
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
    print("Recording... Press and hold 'space' to speak.")
    audio_data = []

    while keyboard.is_pressed('space'):
        chunk = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_data.append(chunk)

    recording = np.concatenate(audio_data, axis=0)
    wavio.write(audio_filename, recording, sample_rate, sampwidth=2)
    print("Recording finished.")
    return audio_filename

while True:
    print("Press and hold 'space' to speak or type 'exit' to stop.")
    keyboard.wait('space')

    # Record voice input when space is held down
    audio_file = record_audio_continuously(sample_rate)

    try:
        # Transcribe the audio file with Whisper
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3-turbo",
                temperature=0,
                response_format="verbose_json",
            )
        user_input = transcription.text
        print("Transcription:", user_input)

        # Add to conversation history
        conversation_history.append({'role': 'user', 'content': user_input})
        save_conversation_history({'role': 'user', 'content': user_input}, history_file_path)

        # Get response from the agent
        agent_response = agent_executor.invoke({"input": user_input, "chat_history": list(conversation_history)})
        ai_response = agent_response['output']
        print('AI:', ai_response)

        conversation_history.append({'role': 'assistant', 'content': ai_response})
        save_conversation_history({'role': 'assistant', 'content': ai_response}, history_file_path)

        # Synthesize and play AI response
        if ai_response:
            asyncio.run(synthesize_and_enqueue_audio(ai_response))

    except Exception as e:
        print("Transcription failed. The audio file has been saved for later use.")
        print("Error details:", e)
