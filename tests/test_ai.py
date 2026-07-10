import json
from unittest.mock import MagicMock, patch

import pytest

from core import ai


def _mock_chat_response(content):
    """Build a fake Groq chat-completion response with the given text content."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


# ---- chat / summarize ---------------------------------------------------

def test_chat_returns_mocked_reply():
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("Hello, how can I help?")
        result = ai.chat("Hi there")
    assert result == "Hello, how can I help?"


def test_summarize_returns_mocked_summary():
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("This is a summary.")
        result = ai.summarize("Some long note content.")
    assert result == "This is a summary."


# ---- generate_flashcards ------------------------------------------------

def test_generate_flashcards_valid_json_parses_to_list_of_dicts():
    payload = json.dumps(
        {"flashcards": [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]}
    )
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response(payload)
        result = ai.generate_flashcards("note content")
    assert result == [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]


def test_generate_flashcards_non_json_response_raises_value_error():
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("this is not json at all")
        with pytest.raises(ValueError):
            ai.generate_flashcards("note content")


def test_generate_flashcards_wrong_top_level_key_raises_value_error():
    # valid JSON, but the model used "cards" instead of "flashcards"
    payload = json.dumps({"cards": [{"question": "Q1", "answer": "A1"}]})
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response(payload)
        with pytest.raises(ValueError):
            ai.generate_flashcards("note content")


# ---- generate_quiz -------------------------------------------------------

def test_generate_quiz_valid_json_parses_to_list_of_dicts():
    payload = json.dumps({"quiz": [{"question": "Q1", "answer": "A1"}]})
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response(payload)
        result = ai.generate_quiz("note content")
    assert result == [{"question": "Q1", "answer": "A1"}]


def test_generate_quiz_non_json_response_raises_value_error():
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("not json")
        with pytest.raises(ValueError):
            ai.generate_quiz("note content")


def test_generate_quiz_wrong_top_level_key_raises_value_error():
    # valid JSON, but the model used "questions" instead of "quiz"
    payload = json.dumps({"questions": [{"question": "Q1", "answer": "A1"}]})
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response(payload)
        with pytest.raises(ValueError):
            ai.generate_quiz("note content")


# ---- suggest_schedule -----------------------------------------------------

def test_suggest_schedule_empty_due_items():
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("No cards due, great job!")
        result = ai.suggest_schedule([])
    assert result == "No cards due, great job!"


def test_suggest_schedule_with_due_items():
    due_items = [
        {
            "question": "What is a cell?",
            "answer": "Basic unit of life",
            "ease_factor": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "due_date": None,
        },
    ]
    with patch("core.ai.client.chat.completions.create") as mock_create:
        mock_create.return_value = _mock_chat_response("Here's your study plan...")
        result = ai.suggest_schedule(due_items)
    assert result == "Here's your study plan..."


# ---- transcribe -----------------------------------------------------------

def test_transcribe_returns_mocked_text(tmp_path):
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"fake wav bytes")

    mock_response = MagicMock()
    mock_response.text = "transcribed text"

    with patch("core.ai.client.audio.transcriptions.create") as mock_create:
        mock_create.return_value = mock_response
        result = ai.transcribe(str(audio_file))

    assert result == "transcribed text"
