import webbrowser
import requests
import glob
import os
import wikipedia
import pywhatkit
import pyautogui
import time
from fuzzywuzzy import fuzz, process

from datetime import datetime
from pytz import timezone, all_timezones
from langchain_core.tools import tool
from googlesearch import search
import unittest
from unittest.mock import patch, MagicMock

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_groq import ChatGroq

# Tool definitions

@tool
def search_wikipedia_and_open(query: str) -> str:
    """
    Searches for a query on Wikipedia and opens the corresponding page in a web browser.

    :param query: The search term to look up on Wikipedia.
    :return: A confirmation message indicating that the Wikipedia page has been opened.
    """
    url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    webbrowser.open(url)
    return f"Wikipedia page opened for query: '{query}'"


@tool
def get_wikipedia_article_summary(query: str) -> str:
    """
    Fetches a summary of a Wikipedia article for a given query.

    :param query: The topic or title of the Wikipedia article to summarize.
    :return: A summary of the Wikipedia article.
    """
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return str(e)


@tool
def play_youtube_video_by_title(video_title: str) -> str:
    """
    Searches for a YouTube video by its title and plays it in the default web browser.

    :param video_title: The title or keywords of the YouTube video to search for and play.
    :return: A confirmation message indicating that the video is playing.
    """
    try:
        # Use pywhatkit to search for and play the video
        pywhatkit.playonyt(video_title)
        return f"Playing video: {video_title}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@tool
def get_weather_forecast(city: str) -> dict:
    """
    Fetches the current weather forecast for a given city using the OpenWeather API.

    :param city: The name of the city to get the weather forecast for.
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
def get_current_datetime(location: str = "UTC") -> str:
    """
    Gets the current date and time for a specific location in a human-readable format.

    :param location: The timezone location for which to get the time. Defaults to "UTC".
    :return: The current date and time at the specified location.
    """
    if location not in all_timezones:
        return f"Error: '{location}' is not a recognized timezone."

    # Set timezone based on location
    tz = timezone(location)
    now = datetime.now(tz)
    current_time = now.strftime("%I:%M:%S %p")  # 12-hour format with AM/PM
    current_date = now.strftime("%A, %B %d, %Y")  # Full weekday, month name, day, and year

    return f"Current Date and Time in {location} = {current_date}, {current_time}"

@tool
def write_to_notepad(content: str) -> str:
    """
    Opens Notepad and writes the given content to it.

    :param content: The text content to be written to Notepad.
    :return: A confirmation message indicating that the content has been written to Notepad.
    """
    try:
        pyautogui.hotkey('win', 'r')
        time.sleep(1)
        pyautogui.write('notepad')
        pyautogui.press('enter')
        time.sleep(2)
        pyautogui.write(content)
        return "Content has been written to Notepad."
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def document_qa_tool(query: str) -> str:
    """
    Answers questions about the documents in the data directory.
    """
    # Initialize the HuggingFace embedding model
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the Groq API LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")

    # Load documents from the specified directory
    loader = DirectoryLoader(
        "data",
        glob="**/*.txt",  # Load only .txt files
        loader_cls=TextLoader,
        use_multithreading=True,
        show_progress=True,
        silent_errors=True,
    )
    documents = loader.load()

    # Create a VectorStoreIndex from the documents
    vector_store = FAISS.from_documents(documents, embed_model)
    retriever = vector_store.as_retriever()

    # Create a RAG chain
    rag_prompt = PromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

class TestTools(unittest.TestCase):

    @patch('webbrowser.open')
    def test_open_wikipedia_search(self, mock_open):
        query = "Python programming"
        expected_url = "https://en.wikipedia.org/wiki/Python_programming"
        result = search_wikipedia_and_open.func(query)
        mock_open.assert_called_with(expected_url)
        self.assertEqual(result, f"Wikipedia page opened for query: '{query}'")

    def test_get_wikipedia_summary(self):
        query = "Python (programming language)"
        summary = get_wikipedia_article_summary.func(query)
        self.assertIsInstance(summary, str)
        self.assertIn("Python", summary)

    @patch('pywhatkit.playonyt')
    def test_play_youtube_video(self, mock_play):
        video_title = "Never Gonna Give You Up"
        result = play_youtube_video_by_title.func(video_title)
        mock_play.assert_called_with(video_title)
        self.assertEqual(result, f"Playing video: {video_title}")

    @patch('requests.get')
    def test_get_current_weather(self, mock_get):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "London",
            "main": {
                "temp": 15,
                "feels_like": 14,
                "humidity": 80
            },
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 5}
        }
        mock_get.return_value = mock_response
        
        city = "London"
        weather = get_weather_forecast.func(city)
        self.assertEqual(weather['city'], 'London')
        self.assertEqual(weather['temperature'], 15)

        # Inform user about API key
        print("\nNote: The get_current_weather tool requires a valid API key from OpenWeather.")

    def test_get_current_time(self):
        location = "America/New_York"
        time_str = get_current_datetime.func(location)
        self.assertIn(location, time_str)
        self.assertIn("Time", time_str)

    @patch('pyautogui.hotkey')
    @patch('pyautogui.write')
    @patch('pyautogui.press')
    @patch('time.sleep')
    def test_write_to_notepad(self, mock_sleep, mock_press, mock_write, mock_hotkey):
        content = "Hello, Notepad!"
        result = write_to_notepad.func(content)
        self.assertEqual(result, "Content has been written to Notepad.")
        self.assertTrue(mock_hotkey.called)
        self.assertTrue(mock_write.called)
        self.assertTrue(mock_press.called)

if __name__ == '__main__':
    unittest.main()