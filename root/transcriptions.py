import sounddevice as sd
import numpy as np
import wavio
import os
import keyboard
import time
import pyautogui
import asyncio
from transformers import pipeline
from groq import Groq
import edge_tts
from pydub import AudioSegment
from pydub.playback import play


# Initialize the Whisper model pipeline for ASR with timestamps enabled
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=0, return_timestamps=True)

# Initialize the Groq model
groq_client = Groq(api_key="Your_API_Key")

# Audio recording parameters
sample_rate = 16000  # Sample rate compatible with Whisper model
audio_filename = "temp_audio.wav"
chunk_duration = 1  # Record in chunks of 1 second

# Define two system prompts
sys_prompt_1 = (
    "Please refine the following transcription by removing all timestamps, "
    "correcting grammar, improving phrasing, and enhancing readability. "
    "Format the final output as an essay with the following structure:\n\n"
    "Title: [Provide a clear and engaging title]\n"
    "Body: Write a cohesive and well-structured essay, divided into paragraphs with a clear introduction, main content, and conclusion.\n\n"
    "{transcription}"
)

sys_prompt_2 = (
    "Please refine the following transcription by removing all timestamps, "
    "correcting grammar, improving phrasing, and enhancing readability. "
    "Format the final output in a log-entry style with the following structure:\n\n"
    "**Date:** [Today's Date]\n"
    "**Time:** [Current Time]\n"
    "**Title:** [A concise title summarizing the log content]\n\n"
    "#### **Log Entry (number):**\n\n"
    "Body of the Log\n"
    "**Signature:** *Name*\n\n"
    "-----\n"
    "{transcription}"
)

async def speak_text(text):
    """Uses edge_tts to speak out the specified text."""
    communicate = edge_tts.Communicate(text, "en-CA-LiamNeural", pitch="-40Hz")
    await communicate.save("notification.mp3")
    audio = AudioSegment.from_mp3("notification.mp3")
    play(audio)

def record_audio_continuously(sample_rate):
    """
    Records audio from the microphone until a key is released, handling longer recordings.
    """
    print("Recording... Press and hold 'Delete' or 'End' to speak.")
    audio_data = []

    while keyboard.is_pressed('delete') or keyboard.is_pressed('end'):
        chunk = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_data.append(chunk)

    recording = np.concatenate(audio_data, axis=0)
    wavio.write(audio_filename, recording, sample_rate, sampwidth=2)
    print("Recording finished.")
    return audio_filename

def correct_grammar(transcription, system_prompt):
    """
    Sends the transcription to the Groq model to correct grammar using the specified system prompt.
    """
    prompt_content = system_prompt.format(transcription=transcription)
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_content}],
        model="llama3-70b-8192"
    )
    return response.choices[0].message.content

while True:
    print("Press and hold 'Delete' for essay format or 'End' for log format.")
    
    # Wait for either 'Delete' or 'End' key to be held down
    while not (keyboard.is_pressed('delete') or keyboard.is_pressed('end')):
        pass  # Wait until one of the keys is pressed

    # Determine which prompt to use based on the key pressed
    if keyboard.is_pressed('delete'):
        system_prompt = sys_prompt_1
    elif keyboard.is_pressed('end'):
        system_prompt = sys_prompt_2

    # Record audio while the chosen key is held down
    audio_file = record_audio_continuously(sample_rate)
    
    try:
        # Transcribe the audio file with Whisper
        result = pipe(audio_file)
        transcription = " ".join([f"{timestamp['text']}" for timestamp in result["chunks"]])
        print("Original Transcription:", transcription)

        # Send transcription to Groq model for grammatical correction with the selected prompt
        corrected_transcription = correct_grammar(transcription, system_prompt)
        print("Formatted Transcription:", corrected_transcription)

        # Notify user before typing
        notification_text = "Preparing to type the formatted text in 5 seconds. Please place your cursor in the desired field."
        print(notification_text)
        
        # Use TTS to read the notification aloud
        asyncio.run(speak_text(notification_text))

        # Wait 5 seconds for the user to place the cursor in the desired field
        time.sleep(5)

        # Type the corrected transcription
        pyautogui.write(corrected_transcription, interval=0.01)

    except Exception as e:
        print("Transcription failed. The audio file has been saved for later use.")
        print("Error details:", e)
