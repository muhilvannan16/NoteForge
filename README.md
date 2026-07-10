<div align="center">

# NoteForge

**An AI-assisted study app — notes, flashcards, quizzes, chat, and spaced repetition.**
**Desktop or web, same tested backend.**

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Tests](https://github.com/muhilvannan16/NoteForge/actions/workflows/tests.yml/badge.svg)
![Desktop](https://img.shields.io/badge/desktop-customtkinter-1f6feb)
![Web](https://img.shields.io/badge/web-streamlit-FF4B4B?logo=streamlit&logoColor=white)
![AI](https://img.shields.io/badge/AI-Groq%20%7C%20Llama%20%7C%20Whisper-6c63ff)
🔗 **[Try NoteForge live](https://noteforge-olasjtojdx8teuotjuiyfp.streamlit.app)**

</div>

---

## Contents

- [Features](#features)
- [Structure](#structure)
- [Desktop app](#desktop-app)
- [Web app](#web-app)
- [Tests](#tests)
- [Tech stack](#tech-stack)

---

## Features

| | |
|---|---|
| 📝 **Notes** | Create, edit, delete; live search across subject, title, *and* content; export to a text file |
| 🤖 **AI Tools** | Summarize a note; generate flashcards (saved to the database, not just displayed); get a suggested study schedule based on what's due |
| 💬 **Chat** | Free-form conversation with the AI — by typing, or by voice (transcribed with Groq's Whisper model) |
| 🔁 **Flashcards** | Spaced repetition review using the real SM-2 algorithm — rate each card Again/Hard/Good/Easy and the schedule adjusts automatically |
| ✅ **Quiz** | AI-generated questions, typed answers, self-graded (Correct/Incorrect), every attempt logged |
| 📊 **Progress** | Accuracy stats overall and per-note, plus recent quiz-attempt history |

---

## Structure

```
noteforge/
├── core/
│   ├── ai.py            # every Groq API call (chat, transcribe, summarize, flashcards, quiz, schedule)
│   ├── database.py      # SQLite: notes, flashcards, quiz_history
│   └── srs.py           # spaced repetition logic (SM-2)
├── gui/
│   └── app.py             # desktop app (customtkinter) — 6 tabs, all wired to core/
├── web_app.py              # web app (Streamlit) — same features, reuses core/ unchanged
├── tests/
│   ├── conftest.py
│   ├── test_database.py
│   ├── test_srs.py
│   └── test_ai.py
├── .github/workflows/
│   └── tests.yml           # runs the test suite automatically on every push
├── .env                     # GROQ_API_KEY=... (desktop only, never committed)
├── requirements.txt         # desktop app dependencies
└── requirements-web.txt     # web app dependencies
```

> `core/` has no knowledge of Tkinter, Streamlit, or any UI framework —
> `gui/app.py` and `web_app.py` both call the exact same tested functions
> from `core/`, never touching SQLite or the Groq API directly.

---

## Desktop app

**Setup**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

**Run**
```bash
python -m gui.app
```

**Build a standalone Windows `.exe`** (no Python installation required to run it):
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name NoteForge gui\app.py
```
The resulting `dist\NoteForge.exe` reads its API key from a `.env` file
placed in the same folder as the executable.

---

## Web app

The web version supports multiple simultaneous visitors, each with:
- their own **private, persistent database** — a random session code maps
  to a separate SQLite file, so visitors never see each other's data
- their own **Groq API key**, entered directly in the browser — the
  host's key is never used

**Setup**
```bash
pip install -r requirements-web.txt
```

**Run**
```bash
streamlit run web_app.py
```

A visitor's session code is shown on first visit — saving it lets them
return later and resume their notes instead of starting fresh.

---

## Tests

**31 automated tests** covering `database.py`, `srs.py`, and `ai.py` (the
last one with mocked API responses — no real network calls or API key
needed to run the suite).

```bash
pip install pytest
pytest tests/ -v
```

Runs automatically on every push via GitHub Actions
(`.github/workflows/tests.yml`).

---

## Tech stack

| Layer | Tools |
|---|---|
| Desktop UI | Python, customtkinter |
| Web UI | Streamlit |
| Data | SQLite |
| AI | Groq API — Llama 3.1 (text), Whisper Large v3 (voice) |
| Testing | pytest, GitHub Actions |
| Packaging | PyInstaller |
