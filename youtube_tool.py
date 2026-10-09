import os
import webbrowser

from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is missing from .env")

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)


def search_youtube(query: str):
    """Search YouTube for a song and return the top results."""

    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=5
    )

    response = request.execute()

    results = []

    for item in response.get("items", []):
        video_id = item["id"]["videoId"]

        results.append({
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })

    return {
        "results": results,
        "instruction": "Choose the single most relevant result and call play_youtube with its URL."
    }


def play_youtube(url: str):
    """Open the selected YouTube video in the default browser."""

    if "youtube.com/watch" not in url:
        return {
            "success": False,
            "error": "Invalid YouTube URL"
        }

    webbrowser.open(url)

    return {
        "success": True,
        "message": "YouTube video opened successfully.",
        "url": url
    }