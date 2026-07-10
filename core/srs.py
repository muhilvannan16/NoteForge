"""
Spaced repetition logic (SM-2 algorithm).

Given a flashcard's current ease factor, interval, and repetition count,
plus a quality score (0-5) from the latest review, compute the next
interval and due date. No Tkinter, no SQLite, no Groq - pure logic.
"""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class ReviewState:
    ease_factor: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    due_date: date = None


def review(state: ReviewState, quality: int) -> ReviewState:
    ease_factor = state.ease_factor
    interval_days = state.interval_days
    repetitions = state.repetitions

    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)   # old ease_factor
        repetitions += 1

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if ease_factor < 1.3:
        ease_factor = 1.3

    due_date = date.today() + timedelta(days=interval_days)

    return ReviewState(
        ease_factor=ease_factor,
        interval_days=interval_days,
        repetitions=repetitions,
        due_date=due_date,
    )
