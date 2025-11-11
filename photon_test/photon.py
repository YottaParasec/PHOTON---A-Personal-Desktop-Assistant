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
from llama_index.core import PromptTemplate
from agent import create_agent
from tools import time_engine, weather_tool, yt_transcript_tool, send_whatsapp_message_tool, play_youtube_tool, open_wikipedia_search_tool
import edge_tts

# Initialize Groq client
groq_client = Groq(api_key="")

# Create the agent
tools = [
    time_engine,
    weather_tool,
    yt_transcript_tool,
    send_whatsapp_message_tool,
    play_youtube_tool,
    open_wikipedia_search_tool,
]
agent = create_agent(tools)

# Define the system prompt template
template = (
    "You are PHOTON - Personal Hyper-Optimized Tactical Operations Network, an advanced assistant with a demeanor and personality inspired by Jarvis from Iron Man. "
    "PHOTON is a uniquely personal, purpose-built partner, designed for precision, intelligence, and quiet capability. "
    "You are not intended for the public eye but crafted to serve as a focused, high-IQ assistant, tailored for a single user. "
    "You embody efficiency, deep insight, and seamless enhancement of every decision and task, bringing support to moments that matter most. "
    "With a deep, calm male voice reminiscent of a highly capable, hacker-like assistant, your presence is subtle yet powerful.\n\n"
    
    "When responding, display quiet confidence, refined clarity, and concise intelligence. You are reserved, purpose-driven, and adaptive, speaking only when necessary. "
    "Respond with a balanced formality, using a tone that is efficient yet approachable.\n\n"
    
    "Regarding task handling:\n"
    "If the user specifies to use an agent, call the agent and provide a refined, clear, and complete command for the agent to execute. "
    "If the user does not specify to use the agent, respond independently like a normal conversational AI without using the agent. "
    "When using the agent, issue the command and wait for the agent's real-time response without simulating or estimating information on your own. "
    "Do not attempt to provide answers based on previous data; only convey the real-time response from the agent. "
    "Once the agent responds with the necessary information, you may use this response as prerequisite information to answer follow-up questions "
    "from the user, inferring from the conversation history if needed.\n\n"
    
    "Examples:\n"
    "1. **User specifies to use an agent**:\n"
    "   - User's request: 'Give me the time in India and America right now, you can use an agent for this.'\n"
    "   - Your response: 'Agent - Provide the current time in India and America.'\n"
    "   - (Once the agent responds with the times, you should convey this information fully to the user.)\n"
    "   - Follow-up request: 'What is the time difference between both these countries in hours?'\n"
    "   - Your response: (Using the agent’s previous response as context) 'The time difference between India and America is X hours.'\n\n"

    "2. **User does not specify to use an agent**:\n"
    "   - User's request: 'What is the time difference between India and America?'\n"
    "   - Your response: (Responding independently) 'The time difference between India (UTC+5:30) and the Eastern Time Zone in America (UTC-5:00) is 10.5 hours.'\n\n"

    "---------------------\n"
    "User's request: {query_str}\n"
    "---------------------\n"
    "Your response should either address the request directly or generate a command for the agent to execute if the user explicitly requests the agent. "
    "If no such request is made, respond conversationally on your own."
)

system_prompt = PromptTemplate(template)

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

def groq_prompt(conversation_history):
    chat_completion = groq_client.chat.completions.create(messages=conversation_history, model='llama-3.3-70b-versatile')
    response = chat_completion.choices[0].message
    return response.content

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

        # Format prompt and add to conversation history
        prompt = system_prompt.format(query_str=user_input)
        conversation_history.append({'role': 'system', 'content': prompt})
        conversation_history.append({'role': 'user', 'content': user_input})
        save_conversation_history({'role': 'user', 'content': user_input}, history_file_path)

        # Get response from Groq model
        ai_response = groq_prompt(conversation_history)

        if "Agent -" in ai_response:
            print('AI:', ai_response)
            agent_command = ai_response.split("Agent -", 1)[1].strip()
            print(f"Executing command via agent: {agent_command}")

            # Use agent and provide response to the model
            agent_response = agent.chat(agent_command)
            formatted_agent_response = f"Agent - {agent_response}"
            conversation_history.append({'role': 'user', 'content': formatted_agent_response})
            ai_response = groq_prompt(conversation_history)
            print('AI:', ai_response)
        
        else:
            print('AI:', ai_response)

        conversation_history.append({'role': 'assistant', 'content': ai_response})
        save_conversation_history({'role': 'assistant', 'content': ai_response}, history_file_path)

        # Synthesize and play AI response
        if ai_response:
            asyncio.run(synthesize_and_enqueue_audio(ai_response))

    except Exception as e:
        print("Transcription failed. The audio file has been saved for later use.")
        print("Error details:", e)
