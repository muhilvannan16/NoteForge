import tempfile
from pathlib import Path

import pytest

from core import database


@pytest.fixture
def temp_db(monkeypatch):
    temp_path = Path(tempfile.mktemp(suffix=".db"))
    monkeypatch.setattr(database, "DB_PATH", temp_path)
    database.init_db()
    yield
    if temp_path.exists():
        temp_path.unlink()


# ---- notes CRUD -------------------------------------------------------

def test_create_and_get_note(temp_db):
    note_id = database.create_note("Bio", "Cells", "Cell content")
    note = database.get_note(note_id)
    assert note is not None
    assert note["subject"] == "Bio"
    assert note["title"] == "Cells"
    assert note["content"] == "Cell content"


def test_get_note_nonexistent_returns_none(temp_db):
    assert database.get_note(999) is None


def test_get_all_notes_reflects_deletions(temp_db):
    n1 = database.create_note("Bio", "Cells", "content 1")
    n2 = database.create_note("Chem", "Acids", "content 2")

    all_notes = database.get_all_notes()
    assert {n["id"] for n in all_notes} == {n1, n2}

    database.delete_note(n1)
    all_notes = database.get_all_notes()
    assert {n["id"] for n in all_notes} == {n2}


def test_update_note(temp_db):
    note_id = database.create_note("Bio", "Cells", "old content")
    database.update_note(note_id, "Biology", "Cell Structure", "new content")
    note = database.get_note(note_id)
    assert note["subject"] == "Biology"
    assert note["title"] == "Cell Structure"
    assert note["content"] == "new content"


def test_delete_note(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    database.delete_note(note_id)
    assert database.get_note(note_id) is None


# ---- flashcards --------------------------------------------------------

def test_create_and_get_flashcard(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    card_id = database.create_flashcard(note_id, "What is a cell?", "Basic unit of life")
    card = database.get_flashcard(card_id)
    assert card is not None
    assert card["note_id"] == note_id
    assert card["question"] == "What is a cell?"
    assert card["answer"] == "Basic unit of life"
    assert card["ease_factor"] == 2.5
    assert card["interval_days"] == 0
    assert card["repetitions"] == 0
    assert card["due_date"] is None


def test_get_flashcard_nonexistent_returns_none(temp_db):
    assert database.get_flashcard(999) is None


def test_get_all_flashcards(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    c1 = database.create_flashcard(note_id, "Q1", "A1")
    c2 = database.create_flashcard(note_id, "Q2", "A2")
    all_cards = database.get_all_flashcards()
    assert {c["id"] for c in all_cards} == {c1, c2}


def test_get_due_flashcards_null_due_date_counts_as_due(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    card_id = database.create_flashcard(note_id, "Q1", "A1")
    due = database.get_due_flashcards()
    assert card_id in {c["id"] for c in due}


def test_get_due_flashcards_future_due_date_excluded(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    card_id = database.create_flashcard(note_id, "Q1", "A1")
    database.update_flashcard_review(card_id, 2.5, 30, 1, "2099-01-01")
    due = database.get_due_flashcards()
    assert card_id not in {c["id"] for c in due}


def test_update_flashcard_review(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    card_id = database.create_flashcard(note_id, "Q1", "A1")
    database.update_flashcard_review(card_id, 2.6, 6, 1, "2026-07-15")
    card = database.get_flashcard(card_id)
    assert card["ease_factor"] == 2.6
    assert card["interval_days"] == 6
    assert card["repetitions"] == 1
    assert card["due_date"] == "2026-07-15"


def test_delete_flashcard(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    card_id = database.create_flashcard(note_id, "Q1", "A1")
    database.delete_flashcard(card_id)
    assert database.get_flashcard(card_id) is None


# ---- quiz history + stats ----------------------------------------------

def test_log_quiz_attempt_and_get_quiz_history(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    database.log_quiz_attempt(note_id, "What is a cell?", "Basic unit of life", "Basic unit of life", True)
    history = database.get_quiz_history()
    assert len(history) == 1
    assert history[0]["note_id"] == note_id
    assert history[0]["question"] == "What is a cell?"
    assert history[0]["correct_answer"] == "Basic unit of life"
    assert history[0]["user_answer"] == "Basic unit of life"
    assert history[0]["is_correct"] == 1


def test_get_quiz_history_filtered_by_note_id(temp_db):
    n1 = database.create_note("Bio", "Cells", "content")
    n2 = database.create_note("Chem", "Acids", "content")
    database.log_quiz_attempt(n1, "Q1", "A1", "A1", True)
    database.log_quiz_attempt(n2, "Q2", "A2", "wrong", False)

    all_history = database.get_quiz_history()
    assert len(all_history) == 2

    n1_history = database.get_quiz_history(n1)
    assert len(n1_history) == 1
    assert n1_history[0]["note_id"] == n1


def test_get_quiz_stats_basic(temp_db):
    note_id = database.create_note("Bio", "Cells", "content")
    database.log_quiz_attempt(note_id, "Q1", "A1", "A1", True)
    database.log_quiz_attempt(note_id, "Q2", "A2", "wrong", False)

    stats = database.get_quiz_stats()
    assert stats["overall"]["total"] == 2
    assert stats["overall"]["correct"] == 1
    assert stats["overall"]["percent"] == 50.0

    by_note = {row["note_id"]: row for row in stats["by_note"]}
    assert by_note[note_id]["subject"] == "Bio"
    assert by_note[note_id]["title"] == "Cells"
    assert by_note[note_id]["correct"] == 1
    assert by_note[note_id]["total"] == 2


def test_get_quiz_stats_deleted_note_edge_case(temp_db):
    # log a quiz attempt against a note_id that was never created,
    # simulating a note that has since been deleted
    deleted_note_id = 12345
    database.log_quiz_attempt(deleted_note_id, "Q1", "A1", "A1", True)

    # also log a real, existing note's attempt so we can confirm both
    # show up together and the totals still add up
    real_note_id = database.create_note("Bio", "Cells", "content")
    database.log_quiz_attempt(real_note_id, "Q2", "A2", "wrong", False)

    stats = database.get_quiz_stats()

    by_note = {row["note_id"]: row for row in stats["by_note"]}
    assert by_note[deleted_note_id]["subject"] == "(Deleted note)"
    assert by_note[deleted_note_id]["correct"] == 1
    assert by_note[deleted_note_id]["total"] == 1

    # the deleted note's attempt must not be silently dropped from the
    # overall total either
    assert stats["overall"]["total"] == 2
    assert stats["overall"]["correct"] == 1

    # sum of all by_note totals must equal the overall total
    assert sum(row["total"] for row in stats["by_note"]) == stats["overall"]["total"]
