"""
All Groq API calls live here: chat, summarize, flashcards, quiz, schedule.

gui/ should only ever call functions from this module - never talk to the
Groq API directly. Keeping every prompt/response in one place is what
prevents the duplicated-logic mess this project is meant to avoid.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def chat(message: str) -> str:
    """Send a single user message and return the assistant's reply."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text using Groq's Whisper model."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, model="whisper-large-v3")
    return response.text


def summarize(note_content: str) -> str:
    """Summarize a note's content concisely, in the model's own words."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Summarize the note the user gives you "
                    "in a few concise sentences that capture the key ideas. Do not "
                    "just rewrite the note in different words line by line - distill "
                    "it down to what actually matters for someone studying it later."
                ),
            },
            {"role": "user", "content": note_content},
        ],
    )
    return response.choices[0].message.content


def generate_flashcards(note_content: str) -> list[dict]:
    """Generate 5 question/answer flashcards from a note's content."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant that turns notes into flashcards. "
                    "Respond only with valid JSON, no other text before or after it. "
                    "The JSON must be an object with exactly this shape: "
                    '{"flashcards": [{"question": "...", "answer": "..."}, ...]} '
                    "containing exactly 5 flashcards."
                ),
            },
            {"role": "user", "content": note_content},
        ],
    )
    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
        return result["flashcards"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Model did not return valid flashcards JSON: {raw!r}") from e


def generate_quiz(note_content: str) -> list[dict]:
    """Generate quiz questions (question/answer pairs) from a note's content."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant that writes quiz questions from notes. "
                    "Respond only with valid JSON, no other text before or after it. "
                    "The JSON must be an object with exactly this shape: "
                    '{"quiz": [{"question": "...", "answer": "..."}, ...]}.'
                ),
            },
            {"role": "user", "content": note_content},
        ],
    )
    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
        return result["quiz"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Model did not return valid quiz JSON: {raw!r}") from e


def suggest_schedule(due_items: list[dict]) -> str:
    """Suggest a short, encouraging study plan given today's due flashcards."""
    if not due_items:
        summary = "There are no flashcards due right now."
    else:
        lines = [f"- {item['question']}" for item in due_items]
        summary = f"{len(due_items)} flashcard(s) are due today:\n" + "\n".join(lines)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a supportive study coach. Given a list of due flashcards, "
                    "write a short, encouraging study plan in plain text - no markdown, "
                    "no JSON. Suggest an order or grouping to tackle them in and keep "
                    "the tone motivating."
                ),
            },
            {"role": "user", "content": summary},
        ],
    )
    return response.choices[0].message.content
