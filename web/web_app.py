"""
NoteForge web companion (Streamlit).

Reuses core/database.py and core/srs.py completely unmodified. Does NOT
import core/ai.py, because that module builds its Groq client from
os.getenv("GROQ_API_KEY") at import time - fine for the desktop app's own
.env, but wrong here since this is a public site where every visitor must
supply their own key. The six ai.py functions are re-implemented below
with the exact same prompts/models/parsing, parameterized by a per-visitor
client instead.

Per-visitor data isolation: each visitor gets a random session code (a
UUID) that maps to its own SQLite file under data/<code>.db. A visitor can
paste back a previously-issued code to resume their notes on a later
visit.
"""

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from groq import Groq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database
from core import srs

st.set_page_config(page_title="NoteForge", layout="wide")


# ---------------------------------------------------------------------------
# AI helpers - same prompts/models/parsing as core/ai.py, but built around a
# per-visitor Groq client instead of a module-level one.
# ---------------------------------------------------------------------------

def get_client() -> Groq:
    return Groq(api_key=st.session_state.get("api_key", ""))


def ai_chat(client: Groq, message: str) -> str:
    """Send a single user message and return the assistant's reply."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content


def ai_transcribe(client: Groq, audio_path: str) -> str:
    """Transcribe an audio file to text using Groq's Whisper model."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, model="whisper-large-v3")
    return response.text


def ai_summarize(client: Groq, note_content: str) -> str:
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


def ai_generate_flashcards(client: Groq, note_content: str) -> list[dict]:
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


def ai_generate_quiz(client: Groq, note_content: str) -> list[dict]:
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


def ai_suggest_schedule(client: Groq, due_items: list[dict]) -> str:
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


def format_qa_list(items: list[dict], header: str) -> str:
    """Format a list of {"question", "answer"} dicts as numbered text."""
    parts = [header]
    for i, item in enumerate(items, start=1):
        parts.append(f"{i}. Q: {item['question']}\n   A: {item['answer']}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Per-visitor session code -> per-visitor SQLite file
# ---------------------------------------------------------------------------

st.title("NoteForge")

if "session_code" not in st.session_state:
    st.session_state["session_code"] = str(uuid.uuid4())

st.subheader("Your session")

resume_input = st.text_input(
    "Have a code from a previous visit? Paste it here to resume your notes.",
    key="resume_code_input",
)
if st.button("Resume this code", key="resume_code_btn"):
    try:
        parsed = uuid.UUID(resume_input.strip())
    except (ValueError, AttributeError, TypeError):
        st.error("That doesn't look like a valid code - please double-check and try again.")
    else:
        st.session_state["session_code"] = str(parsed)
        st.rerun()

session_code = st.session_state["session_code"]

st.success("Save this code to access your notes again later:")
st.code(session_code)
st.caption(
    "Your notes persist as long as this app stays active. If they ever "
    "disappear, that means the app restarted — this isn't guaranteed "
    "forever on free hosting, so don't rely on it for anything critical."
)

# Monkeypatch DB_PATH to a per-visitor file before ANY database call, once
# per script run, before any tab logic runs.
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
database.DB_PATH = data_dir / f"{session_code}.db"
database.init_db()


# ---------------------------------------------------------------------------
# Per-visitor Groq API key
# ---------------------------------------------------------------------------

st.subheader("Your Groq API key")

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

api_key = st.text_input(
    "Your Groq API key",
    type="password",
    key="api_key",
    help="Only used for your own requests - never shared with other visitors.",
)
has_key = bool(api_key.strip())

if not has_key:
    st.warning(
        "Enter your Groq API key above to enable AI features "
        "(Summarize, Flashcards, Quiz, Chat, Suggest Schedule)."
    )


# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------

if "selected_note_id" not in st.session_state:
    st.session_state["selected_note_id"] = None


tab_notes, tab_ai, tab_chat, tab_flashcards, tab_quiz, tab_progress = st.tabs(
    ["Notes", "AI Tools", "Chat", "Flashcards", "Quiz", "Progress"]
)


# ---------------------------------------------------------------------------
# Notes tab
# ---------------------------------------------------------------------------

with tab_notes:
    list_col, editor_col = st.columns([1, 2])

    with list_col:
        st.subheader("Your notes")

        search_term = st.text_input("Search notes...", key="notes_search_input").strip().lower()

        all_notes = database.get_all_notes()
        if search_term:
            visible_notes = [
                n
                for n in all_notes
                if search_term in n["subject"].lower()
                or search_term in n["title"].lower()
                or search_term in (n["content"] or "").lower()
            ]
        else:
            visible_notes = all_notes

        for n in visible_notes:
            if st.button(f"{n['subject']} — {n['title']}", key=f"note_select_{n['id']}"):
                st.session_state["selected_note_id"] = n["id"]
                st.rerun()

    with editor_col:
        st.subheader("Editor")

        selected_id = st.session_state.get("selected_note_id")
        note = database.get_note(selected_id) if selected_id is not None else None
        if selected_id is not None and note is None:
            # the selected note no longer exists (e.g. deleted elsewhere)
            st.session_state["selected_note_id"] = None
            selected_id = None

        subject = st.text_input(
            "Subject", value=note["subject"] if note else "", key=f"editor_subject_{selected_id}"
        )
        title = st.text_input(
            "Title", value=note["title"] if note else "", key=f"editor_title_{selected_id}"
        )
        content = st.text_area(
            "Content",
            value=(note["content"] or "") if note else "",
            key=f"editor_content_{selected_id}",
            height=250,
        )

        new_col, save_col, delete_col = st.columns(3)
        with new_col:
            if st.button("New", key="notes_new_btn"):
                st.session_state["selected_note_id"] = None
                for k in ("editor_subject_None", "editor_title_None", "editor_content_None"):
                    st.session_state.pop(k, None)
                st.rerun()
        with save_col:
            if st.button("Save", key="notes_save_btn"):
                if selected_id is None:
                    new_id = database.create_note(subject, title, content)
                    st.session_state["selected_note_id"] = new_id
                else:
                    database.update_note(selected_id, subject, title, content)
                st.rerun()
        with delete_col:
            if st.button("Delete", key="notes_delete_btn", disabled=selected_id is None):
                database.delete_note(selected_id)
                st.session_state["selected_note_id"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# AI Tools tab
# ---------------------------------------------------------------------------

with tab_ai:
    st.subheader("AI Tools")

    selected_id = st.session_state.get("selected_note_id")
    note = database.get_note(selected_id) if selected_id is not None else None
    note_content = note["content"] if note else ""

    if "ai_output" not in st.session_state:
        st.session_state["ai_output"] = ""

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Summarize", key="ai_summarize_btn", disabled=not has_key):
            if not (note_content or "").strip():
                st.session_state["ai_output"] = "No content to summarize."
            else:
                client = get_client()
                try:
                    st.session_state["ai_output"] = ai_summarize(client, note_content)
                except Exception as e:
                    st.session_state["ai_output"] = f"Error: {e}"
            st.rerun()

    with col2:
        if st.button("Generate Flashcards", key="ai_flashcards_btn", disabled=not has_key):
            if selected_id is None:
                st.session_state["ai_output"] = "Save this note before generating flashcards."
            else:
                client = get_client()
                try:
                    flashcards = ai_generate_flashcards(client, note_content)
                    for card in flashcards:
                        database.create_flashcard(selected_id, card["question"], card["answer"])
                    st.session_state["ai_output"] = format_qa_list(
                        flashcards, f"{len(flashcards)} flashcards created:"
                    )
                except Exception as e:
                    st.session_state["ai_output"] = f"Error: {e}"
            st.rerun()

    with col3:
        if st.button("Suggest Schedule", key="ai_schedule_btn", disabled=not has_key):
            client = get_client()
            try:
                due_items = database.get_due_flashcards()
                st.session_state["ai_output"] = ai_suggest_schedule(client, due_items)
            except Exception as e:
                st.session_state["ai_output"] = f"Error: {e}"
            st.rerun()

    if st.session_state["ai_output"]:
        st.text(st.session_state["ai_output"])


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

with tab_chat:
    st.subheader("Chat")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_input_key_version" not in st.session_state:
        st.session_state["chat_input_key_version"] = 0
    if "last_audio_hash" not in st.session_state:
        st.session_state["last_audio_hash"] = None

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    audio_value = st.audio_input("Record a voice message", key="chat_audio_input")
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state["last_audio_hash"]:
            st.session_state["last_audio_hash"] = audio_hash
            if not has_key:
                st.warning("Enter your Groq API key above to transcribe voice messages.")
            else:
                tmp_path = None
                transcribed = None
                try:
                    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    with open(tmp_path, "wb") as f:
                        f.write(audio_bytes)
                    client = get_client()
                    transcribed = ai_transcribe(client, tmp_path)
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                if transcribed is not None:
                    message_key = f"chat_message_input_{st.session_state['chat_input_key_version']}"
                    st.session_state[message_key] = transcribed
                    st.rerun()

    message_key = f"chat_message_input_{st.session_state['chat_input_key_version']}"
    message = st.text_input("Type a message...", key=message_key)

    if st.button("Send", key="chat_send_btn", disabled=not has_key or not message.strip()):
        st.session_state["chat_history"].append({"role": "user", "content": message})
        client = get_client()
        try:
            reply = ai_chat(client, message)
        except Exception as e:
            reply = f"Error: {e}"
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.session_state["chat_input_key_version"] += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Flashcards tab
# ---------------------------------------------------------------------------

with tab_flashcards:
    st.subheader("Flashcards")

    if "flashcard_queue" not in st.session_state:
        st.session_state["flashcard_queue"] = database.get_due_flashcards()
        st.session_state["flashcard_index"] = 0
        st.session_state["flashcard_revealed"] = False

    if st.button("Refresh", key="flashcards_refresh_btn"):
        st.session_state["flashcard_queue"] = database.get_due_flashcards()
        st.session_state["flashcard_index"] = 0
        st.session_state["flashcard_revealed"] = False
        st.rerun()

    queue = st.session_state["flashcard_queue"]
    idx = st.session_state["flashcard_index"]

    if not queue or idx >= len(queue):
        st.info("No cards due right now!")
    else:
        card = queue[idx]
        st.write(f"Card {idx + 1} of {len(queue)}")

        if st.session_state["flashcard_revealed"]:
            st.text(f"{card['question']}\n\n---\n\n{card['answer']}")

            again_col, hard_col, good_col, easy_col = st.columns(4)
            rating_buttons = [
                (again_col, "Again", 1),
                (hard_col, "Hard", 3),
                (good_col, "Good", 4),
                (easy_col, "Easy", 5),
            ]
            for col, label, quality in rating_buttons:
                with col:
                    if st.button(label, key=f"flashcard_rate_{label}_{idx}"):
                        state = srs.ReviewState(
                            ease_factor=card["ease_factor"],
                            interval_days=card["interval_days"],
                            repetitions=card["repetitions"],
                        )
                        new_state = srs.review(state, quality)
                        database.update_flashcard_review(
                            card["id"],
                            new_state.ease_factor,
                            new_state.interval_days,
                            new_state.repetitions,
                            new_state.due_date.isoformat(),
                        )
                        st.session_state["flashcard_index"] += 1
                        st.session_state["flashcard_revealed"] = False
                        st.rerun()
        else:
            st.text(card["question"])
            if st.button("Show Answer", key=f"flashcard_show_answer_{idx}"):
                st.session_state["flashcard_revealed"] = True
                st.rerun()


# ---------------------------------------------------------------------------
# Quiz tab
# ---------------------------------------------------------------------------

with tab_quiz:
    st.subheader("Quiz")

    if "current_quiz" not in st.session_state:
        st.session_state["current_quiz"] = []
        st.session_state["quiz_index"] = 0
        st.session_state["quiz_note_id"] = None
        st.session_state["quiz_score"] = {"correct": 0, "total": 0}
        st.session_state["quiz_revealed"] = False
        st.session_state["quiz_submitted_answer"] = ""

    selected_id = st.session_state.get("selected_note_id")

    if st.button("Generate Quiz", key="quiz_generate_btn", disabled=not has_key):
        if selected_id is None:
            st.error("Save this note before generating a quiz.")
        else:
            note = database.get_note(selected_id)
            client = get_client()
            try:
                quiz = ai_generate_quiz(client, note["content"])
                st.session_state["current_quiz"] = quiz
                st.session_state["quiz_note_id"] = selected_id
                st.session_state["quiz_index"] = 0
                st.session_state["quiz_score"] = {"correct": 0, "total": 0}
                st.session_state["quiz_revealed"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    quiz = st.session_state["current_quiz"]
    qidx = st.session_state["quiz_index"]
    score = st.session_state["quiz_score"]

    if not quiz or qidx >= len(quiz):
        if score["total"]:
            st.info(f"Quiz complete! {score['correct']}/{score['total']} correct")
        else:
            st.info("Generate a quiz to get started.")
    else:
        question = quiz[qidx]
        st.write(f"Question {qidx + 1} of {len(quiz)} — {score['correct']}/{score['total']} correct so far")

        if not st.session_state["quiz_revealed"]:
            st.text(question["question"])
            answer = st.text_input("Your answer", key=f"quiz_answer_{qidx}")
            if st.button("Submit Answer", key=f"quiz_submit_{qidx}"):
                st.session_state["quiz_submitted_answer"] = answer
                st.session_state["quiz_revealed"] = True
                st.rerun()
        else:
            st.text(
                f"Q: {question['question']}\n\n"
                f"Your answer: {st.session_state['quiz_submitted_answer']}\n"
                f"Correct answer: {question['answer']}"
            )
            correct_col, incorrect_col = st.columns(2)
            with correct_col:
                if st.button("Correct", key=f"quiz_correct_{qidx}"):
                    database.log_quiz_attempt(
                        st.session_state["quiz_note_id"],
                        question["question"],
                        question["answer"],
                        st.session_state["quiz_submitted_answer"],
                        True,
                    )
                    st.session_state["quiz_score"]["total"] += 1
                    st.session_state["quiz_score"]["correct"] += 1
                    st.session_state["quiz_index"] += 1
                    st.session_state["quiz_revealed"] = False
                    st.rerun()
            with incorrect_col:
                if st.button("Incorrect", key=f"quiz_incorrect_{qidx}"):
                    database.log_quiz_attempt(
                        st.session_state["quiz_note_id"],
                        question["question"],
                        question["answer"],
                        st.session_state["quiz_submitted_answer"],
                        False,
                    )
                    st.session_state["quiz_score"]["total"] += 1
                    st.session_state["quiz_index"] += 1
                    st.session_state["quiz_revealed"] = False
                    st.rerun()


# ---------------------------------------------------------------------------
# Progress tab
# ---------------------------------------------------------------------------

with tab_progress:
    st.subheader("Progress")

    if st.button("Refresh", key="progress_refresh_btn"):
        st.rerun()

    stats = database.get_quiz_stats()
    overall = stats["overall"]
    if overall["total"]:
        st.write(f"Overall: {overall['correct']}/{overall['total']} correct ({overall['percent']}%)")
    else:
        st.write("No quiz attempts yet.")

    total_flashcards = len(database.get_all_flashcards())
    due_flashcards = len(database.get_due_flashcards())
    st.write(f"{total_flashcards} flashcards total, {due_flashcards} due now")

    st.markdown("**By note**")
    if stats["by_note"]:
        for row in stats["by_note"]:
            st.write(f"{row['subject']} — {row['title']}: {row['correct']}/{row['total']} correct ({row['percent']}%)")
    else:
        st.write("Nothing here yet.")

    st.markdown("**Recent attempts**")
    recent = database.get_quiz_history()[:10]
    if recent:
        for attempt in recent:
            mark = "✓" if attempt["is_correct"] else "✗"
            st.write(
                f"{mark} {attempt['question']} — you said: {attempt['user_answer']} "
                f"(correct: {attempt['correct_answer']})"
            )
    else:
        st.write("Nothing here yet.")
