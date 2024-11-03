import os
import requests
import json
import time
import pyautogui
import webbrowser
from datetime import datetime
from pytz import timezone, all_timezones
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from email.mime.text import MIMEText
import smtplib
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama  # Assuming you're using Ollama LLM for summarization
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import pywhatkit
from fuzzywuzzy import fuzz, process


def open_wikipedia_search(query: str) -> str:
    """
    Opens Wikipedia in the default web browser and searches for the provided query.

    :param query: The search term to look up on Wikipedia.
    :return: Confirmation message of action taken.
    """
    url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    webbrowser.open(url)
    return f"Wikipedia page opened for query: '{query}'"

# Wrap the function with FunctionTool
open_wikipedia_search_tool = FunctionTool.from_defaults(
    open_wikipedia_search,
    name="wikipedia_search",
    description="Open Wikipedia and search for a query"
    )


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

# Wrap the function with FunctionTool for integration
play_youtube_tool = FunctionTool.from_defaults(
    play_youtube_video,
    name="play_youtube_video",
    description="This function takes the title of a YouTube video as input and plays the video in the browser. "
                "The input should be a string describing the video's title or keywords."
)

# Dictionary of contacts with names as keys and phone numbers as values
contacts = {
    #Add your contacts here
}

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

# Creating the FunctionTool wrapper
send_whatsapp_message_tool = FunctionTool.from_defaults(
    send_whatsapp_message,
    name="send_whatsapp_message",
    description="Sends a WhatsApp message to a specified contact. "
                "If the contact name is not found directly, it will use fuzzy matching to find the closest match."
)


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

# Wrap the function with FunctionTool
yt_transcript_tool = FunctionTool.from_defaults(
    get_video_transcript,
    name="youtube_transcript",
    description="Fetches the transcript of a YouTube video by video ID and writes it to a file."
)

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

# Wrap the weather function as a FunctionTool
weather_tool = FunctionTool.from_defaults(get_current_weather, name="weather_data")

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


# Create the time tool for the ReAct Agent
time_engine = FunctionTool.from_defaults(get_current_time, name="get_time")
