"""
SQLite setup + queries for NoteForge.

gui/ and other modules should only ever read/write notes through the
functions here - never open sqlite3 connections elsewhere in the app.
"""

import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "noteforge.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the database file, notes, flashcards, and quiz_history tables if they don't exist yet."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS flashcards (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id       INTEGER,
            question      TEXT NOT NULL,
            answer        TEXT NOT NULL,
            ease_factor   REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 0,
            repetitions   INTEGER NOT NULL DEFAULT 0,
            due_date      TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id        INTEGER,
            question       TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            user_answer    TEXT NOT NULL,
            is_correct     INTEGER NOT NULL,
            attempted_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_note(subject: str, title: str, content: str) -> int:
    """Insert a new note, return its new id."""
    conn = get_connection()
    cursor = conn.execute("INSERT INTO notes (subject, title, content) VALUES (?, ?, ?)", (subject, title, content))
    conn.commit()
    conn.close()
    return cursor.lastrowid

def get_all_notes() -> list[dict]:
    """Return every note as a list of dicts, most recently added first."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM notes ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_note(note_id: int) -> dict | None:
    """Return one note by id, or None if it doesn't exist."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)

def update_note(note_id: int, subject: str, title: str, content: str) -> None:
    """Overwrite an existing note's fields."""
    conn = get_connection()
    conn.execute("UPDATE notes SET subject = ?, title = ?, content = ? WHERE id = ?", (subject, title, content, note_id))
    conn.commit()
    conn.close()

def delete_note(note_id: int) -> None:
    """Remove a note by id."""
    conn = get_connection()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

def create_flashcard(note_id: int, question: str, answer: str) -> int:
    """Insert a new flashcard, return its new id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO flashcards (note_id, question, answer) VALUES (?, ?, ?)",
        (note_id, question, answer),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid

def get_flashcard(flashcard_id: int) -> dict | None:
    """Return one flashcard by id, or None if it doesn't exist."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM flashcards WHERE id = ?", (flashcard_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)

def get_all_flashcards() -> list[dict]:
    """Return every flashcard as a list of dicts, most recently added first."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM flashcards ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_due_flashcards() -> list[dict]:
    """Return flashcards with no due_date yet or a due_date on/before today."""
    conn = get_connection()
    today = date.today().isoformat()
    rows = conn.execute(
        "SELECT * FROM flashcards WHERE due_date IS NULL OR due_date <= ? ORDER BY id",
        (today,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_flashcard_review(
    flashcard_id: int,
    ease_factor: float,
    interval_days: int,
    repetitions: int,
    due_date: str,
) -> None:
    """Overwrite a flashcard's review fields after it has been reviewed."""
    conn = get_connection()
    conn.execute(
        """
        UPDATE flashcards
        SET ease_factor = ?, interval_days = ?, repetitions = ?, due_date = ?
        WHERE id = ?
        """,
        (ease_factor, interval_days, repetitions, due_date, flashcard_id),
    )
    conn.commit()
    conn.close()

def delete_flashcard(flashcard_id: int) -> None:
    """Remove a flashcard by id."""
    conn = get_connection()
    conn.execute("DELETE FROM flashcards WHERE id = ?", (flashcard_id,))
    conn.commit()
    conn.close()

def log_quiz_attempt(
    note_id: int,
    question: str,
    correct_answer: str,
    user_answer: str,
    is_correct: bool,
) -> int:
    """Insert one quiz attempt, return its new id."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO quiz_history (note_id, question, correct_answer, user_answer, is_correct, attempted_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (note_id, question, correct_answer, user_answer, 1 if is_correct else 0, date.today().isoformat()),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid

def get_quiz_history(note_id: int | None = None) -> list[dict]:
    """Return all quiz attempts, or only those for note_id if one is given."""
    conn = get_connection()
    if note_id is None:
        rows = conn.execute("SELECT * FROM quiz_history ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM quiz_history WHERE note_id = ? ORDER BY id DESC",
            (note_id,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_quiz_stats() -> dict:
    """Return overall and per-note quiz accuracy statistics."""
    conn = get_connection()
    overall_row = conn.execute(
        "SELECT COUNT(*) as total, SUM(is_correct) as correct FROM quiz_history"
    ).fetchone()
    total = overall_row["total"] or 0
    correct = overall_row["correct"] or 0
    overall_percent = round((correct / total) * 100, 1) if total else 0.0
    by_note_rows = conn.execute(
        """
        SELECT quiz_history.note_id AS note_id,
               COALESCE(notes.subject, '(Deleted note)') AS subject,
               COALESCE(notes.title, '') AS title,
               COUNT(*) AS total,
               SUM(quiz_history.is_correct) AS correct
        FROM quiz_history
        LEFT JOIN notes ON quiz_history.note_id = notes.id
        GROUP BY quiz_history.note_id
        ORDER BY total DESC
        """
    ).fetchall()
    conn.close()
    by_note = []
    for row in by_note_rows:
        row_total = row["total"]
        row_correct = row["correct"] or 0
        percent = round((row_correct / row_total) * 100, 1) if row_total else 0.0
        by_note.append({
            "note_id": row["note_id"],
            "subject": row["subject"],
            "title": row["title"],
            "correct": row_correct,
            "total": row_total,
            "percent": percent,
        })
    return {
        "overall": {"correct": correct, "total": total, "percent": overall_percent},
        "by_note": by_note,
    }

if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
