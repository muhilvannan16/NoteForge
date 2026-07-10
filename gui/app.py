"""
customtkinter window for NoteForge.

This module only handles UI: layout, widgets, event handlers. It imports
functions from core/ (database, ai, srs) and never touches SQLite or the
Groq API directly.
"""

import os
import tempfile
import threading
import wave

import customtkinter as ctk
import numpy as np
import sounddevice as sd

from core import ai
from core import database
from core import srs
import re

def clean_markdown_artifacts(text):
    """Strip markdown escape backslashes and bold ** markers for plain-text display."""
    text = re.sub(r'\\([*_#])', r'\1', text)      # \* -> *, \_ -> _, \# -> #
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** -> bold (unrendered, but no more junk)
    return text

def _record_audio(duration=5, samplerate=16000):
    """Blocking: records `duration` seconds of mono audio, returns a WAV file path."""
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    int16_audio = (recording * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(int16_audio.tobytes())
    return path


class NoteForgeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NoteForge")
        self.geometry("900x600")

        # sidebar: a scrollable frame on the left
        self.sidebar = ctk.CTkScrollableFrame(self, width=220, label_text="Notes")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *args: self.refresh_notes_list())

        self.search_entry = ctk.CTkEntry(
            self.sidebar, placeholder_text="Search notes...", textvariable=self.search_var
        )
        self.search_entry.pack(fill="x", padx=5, pady=(0, 5))

        # note buttons live in their own sub-frame so refresh_notes_list()
        # can clear/rebuild them without ever touching the search entry
        self.notes_list_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.notes_list_frame.pack(fill="both", expand=True)

        # main area: a tabview on the right
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.tabview.add("Notes")
        self.tabview.add("AI Tools")
        self.tabview.add("Chat")
        self.tabview.add("Flashcards")
        self.tabview.add("Quiz")
        self.tabview.add("Progress")

        self.tabview.set("Notes")

        # Notes tab: subject/title entries, content textbox, action buttons
        notes_tab = self.tabview.tab("Notes")

        self.subject_entry = ctk.CTkEntry(notes_tab, placeholder_text="Subject")
        self.subject_entry.pack(fill="x", padx=10, pady=(10, 5))

        self.title_entry = ctk.CTkEntry(notes_tab, placeholder_text="Title")
        self.title_entry.pack(fill="x", padx=10, pady=5)

        self.content_textbox = ctk.CTkTextbox(notes_tab)
        self.content_textbox.pack(fill="both", expand=True, padx=10, pady=5)

        button_row = ctk.CTkFrame(notes_tab, fg_color="transparent")
        button_row.pack(fill="x", padx=10, pady=(5, 10))

        self.new_button = ctk.CTkButton(button_row, text="New", command=self.new_note)
        self.new_button.pack(side="left", padx=(0, 5))

        self.save_button = ctk.CTkButton(button_row, text="Save", command=self.save_note)
        self.save_button.pack(side="left", padx=5)

        self.delete_button = ctk.CTkButton(button_row, text="Delete", command=self.delete_note)
        self.delete_button.pack(side="left", padx=5)

        # AI Tools tab: output box + one button per AI action
        ai_tab = self.tabview.tab("AI Tools")

        self.ai_output = ctk.CTkTextbox(ai_tab)
        self.ai_output.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        ai_button_row = ctk.CTkFrame(ai_tab, fg_color="transparent")
        ai_button_row.pack(fill="x", padx=10, pady=(5, 10))

        self.summarize_button = ctk.CTkButton(ai_button_row, text="Summarize", command=self.run_summarize)
        self.summarize_button.pack(side="left", padx=(0, 5))

        self.flashcards_button = ctk.CTkButton(
            ai_button_row, text="Generate Flashcards", command=self.run_generate_flashcards
        )
        self.flashcards_button.pack(side="left", padx=5)

        self.schedule_button = ctk.CTkButton(
            ai_button_row, text="Suggest Schedule", command=self.run_suggest_schedule
        )
        self.schedule_button.pack(side="left", padx=5)

        # Chat tab: free-form conversation with ai.chat(), plus voice input
        chat_tab = self.tabview.tab("Chat")

        self.chat_display = ctk.CTkTextbox(chat_tab)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        self.chat_display.configure(state="disabled")

        chat_entry_row = ctk.CTkFrame(chat_tab, fg_color="transparent")
        chat_entry_row.pack(fill="x", padx=10, pady=(5, 10))

        self.chat_entry = ctk.CTkEntry(chat_entry_row, placeholder_text="Type a message...")
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.chat_send_button = ctk.CTkButton(chat_entry_row, text="Send", command=self.send_chat_message)
        self.chat_send_button.pack(side="left", padx=5)

        self.chat_record_button = ctk.CTkButton(
            chat_entry_row, text="Record", command=self.record_voice_message
        )
        self.chat_record_button.pack(side="left", padx=5)

        # Flashcards tab: spaced-repetition review interface
        flashcards_tab = self.tabview.tab("Flashcards")

        self.refresh_flashcards_button = ctk.CTkButton(
            flashcards_tab, text="Refresh", command=self.load_due_flashcards
        )
        self.refresh_flashcards_button.pack(anchor="ne", padx=10, pady=(10, 0))

        self.flashcard_progress = ctk.CTkLabel(flashcards_tab, text="")
        self.flashcard_progress.pack(padx=10, pady=(5, 5))

        self.flashcard_display = ctk.CTkTextbox(flashcards_tab)
        self.flashcard_display.pack(fill="both", expand=True, padx=10, pady=5)
        self.flashcard_display.configure(state="disabled")

        self.show_answer_button = ctk.CTkButton(flashcards_tab, text="Show Answer", command=self.show_answer)
        self.show_answer_button.pack(padx=10, pady=(5, 5))

        rating_row = ctk.CTkFrame(flashcards_tab, fg_color="transparent")
        rating_row.pack(fill="x", padx=10, pady=(5, 10))

        self.again_button = ctk.CTkButton(rating_row, text="Again", command=lambda: self.rate_card(1))
        self.again_button.pack(side="left", padx=(0, 5))

        self.hard_button = ctk.CTkButton(rating_row, text="Hard", command=lambda: self.rate_card(3))
        self.hard_button.pack(side="left", padx=5)

        self.good_button = ctk.CTkButton(rating_row, text="Good", command=lambda: self.rate_card(4))
        self.good_button.pack(side="left", padx=5)

        self.easy_button = ctk.CTkButton(rating_row, text="Easy", command=lambda: self.rate_card(5))
        self.easy_button.pack(side="left", padx=5)

        # Quiz tab: AI-generated quiz with self-graded scoring
        quiz_tab = self.tabview.tab("Quiz")

        self.generate_quiz_button = ctk.CTkButton(quiz_tab, text="Generate Quiz", command=self.run_generate_quiz)
        self.generate_quiz_button.pack(padx=10, pady=(10, 5))

        self.quiz_progress = ctk.CTkLabel(quiz_tab, text="")
        self.quiz_progress.pack(padx=10, pady=(5, 5))

        self.quiz_question_display = ctk.CTkTextbox(quiz_tab)
        self.quiz_question_display.pack(fill="both", expand=True, padx=10, pady=5)
        self.quiz_question_display.configure(state="disabled")

        self.quiz_answer_entry = ctk.CTkEntry(quiz_tab, placeholder_text="Your answer")
        self.quiz_answer_entry.pack(fill="x", padx=10, pady=5)

        self.quiz_submit_button = ctk.CTkButton(quiz_tab, text="Submit Answer", command=self.submit_quiz_answer)
        self.quiz_submit_button.pack(padx=10, pady=(5, 5))

        quiz_grade_row = ctk.CTkFrame(quiz_tab, fg_color="transparent")
        quiz_grade_row.pack(fill="x", padx=10, pady=(5, 10))

        self.quiz_correct_button = ctk.CTkButton(
            quiz_grade_row, text="Correct", command=lambda: self.grade_quiz_answer(True)
        )
        self.quiz_correct_button.pack(side="left", padx=(0, 5))

        self.quiz_incorrect_button = ctk.CTkButton(
            quiz_grade_row, text="Incorrect", command=lambda: self.grade_quiz_answer(False)
        )
        self.quiz_incorrect_button.pack(side="left", padx=5)

        # Progress tab: quiz accuracy + flashcard stats, read-only
        progress_tab = self.tabview.tab("Progress")

        self.refresh_progress_button = ctk.CTkButton(
            progress_tab, text="Refresh", command=self.load_progress_stats
        )
        self.refresh_progress_button.pack(anchor="ne", padx=10, pady=(10, 0))

        self.progress_overall_label = ctk.CTkLabel(progress_tab, text="")
        self.progress_overall_label.pack(padx=10, pady=(5, 5), anchor="w")

        self.progress_flashcards_label = ctk.CTkLabel(progress_tab, text="")
        self.progress_flashcards_label.pack(padx=10, pady=(0, 10), anchor="w")

        self.progress_by_note_frame = ctk.CTkScrollableFrame(progress_tab, label_text="By Note")
        self.progress_by_note_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.progress_recent_frame = ctk.CTkScrollableFrame(progress_tab, label_text="Recent Attempts")
        self.progress_recent_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.current_note_id = None
        self.refresh_notes_list()   # ← populate the sidebar for the first time

        self.due_flashcards = []
        self.current_card_index = 0
        self.answer_visible = False
        self.load_due_flashcards()

        self.current_quiz = []
        self.quiz_index = 0
        self.quiz_note_id = None
        self.quiz_score = {"correct": 0, "total": 0}
        self.quiz_submitted_answer = ""
        self.show_current_quiz_question()   # ← start the Quiz tab in its empty/no-quiz state

        self.load_progress_stats()   # ← populate the Progress tab for the first time

    def refresh_notes_list(self):
        """Rebuild the sidebar's note buttons from the database, filtered by search."""
        for widget in self.notes_list_frame.winfo_children():
            widget.destroy()

        search_term = self.search_var.get().lower()

        for note in database.get_all_notes():
            if search_term and not (
                search_term in note["subject"].lower()
                or search_term in note["title"].lower()
                or search_term in (note["content"] or "").lower()
            ):
                continue

            button = ctk.CTkButton(
                self.notes_list_frame,
                text=f"{note['subject']} — {note['title']}",
                command=lambda note_id=note["id"]: self.load_note(note_id),
            )
            button.pack(fill="x", padx=5, pady=2)

    def load_note(self, note_id):
        """Load one note's fields into the editor."""
        note = database.get_note(note_id)
        if note is None:
            return

        self.current_note_id = note_id

        self.subject_entry.delete(0, "end")
        self.subject_entry.insert(0, note["subject"])

        self.title_entry.delete(0, "end")
        self.title_entry.insert(0, note["title"])

        self.content_textbox.delete("1.0", "end")
        self.content_textbox.insert("1.0", note["content"] or "")

    def new_note(self):
        """Clear the editor fields so Save creates a brand-new note."""
        self.current_note_id = None
        self.subject_entry.delete(0, "end")
        self.title_entry.delete(0, "end")
        self.content_textbox.delete("1.0", "end")

    def save_note(self):
        """Save the editor fields as a new note, or update the loaded one."""
        subject = self.subject_entry.get()
        title = self.title_entry.get()
        content = self.content_textbox.get("1.0", "end-1c")

        if self.current_note_id is None:
            self.current_note_id = database.create_note(subject, title, content)
        else:
            database.update_note(self.current_note_id, subject, title, content)

        self.refresh_notes_list()

    def delete_note(self):
        """Delete the currently loaded note, if any, and reset the editor."""
        if self.current_note_id is not None:
            database.delete_note(self.current_note_id)
            self.new_note()
            self.refresh_notes_list()

    # ---- AI Tools helpers -------------------------------------------------

    def _show_ai_output(self, text):
        text = clean_markdown_artifacts(text)
        self.ai_output.delete("1.0", "end")
        self.ai_output.insert("1.0", text)

    def _format_qa_list(self, items, header):
        """Format a list of {"question", "answer"} dicts as numbered text."""
        parts = [header]
        for i, item in enumerate(items, start=1):
            parts.append(f"{i}. Q: {item['question']}\n   A: {item['answer']}")
        return "\n\n".join(parts) + "\n"

    # ---- Summarize ----------------------------------------------------

    def run_summarize(self):
        content = self.content_textbox.get("1.0", "end-1c")
        if not content.strip():
            self._show_ai_output("No content to summarize.")
            return
        self.summarize_button.configure(state="disabled", text="Summarizing...")
        threading.Thread(target=self._summarize_worker, args=(content,), daemon=True).start()

    def _summarize_worker(self, content):
        try:
            result = ai.summarize(content)
        except Exception as e:
            result = f"Error: {e}"
        self.after(0, self._summarize_done, result)

    def _summarize_done(self, result):
        self._show_ai_output(result)
        self.summarize_button.configure(state="normal", text="Summarize")

    # ---- Generate Flashcards -------------------------------------------

    def run_generate_flashcards(self):
        if self.current_note_id is None:
            self._show_ai_output("Save this note before generating flashcards.")
            return
        content = self.content_textbox.get("1.0", "end-1c")
        self.flashcards_button.configure(state="disabled", text="Generating...")
        threading.Thread(target=self._generate_flashcards_worker, args=(content,), daemon=True).start()

    def _generate_flashcards_worker(self, content):
        try:
            flashcards = ai.generate_flashcards(content)
            for card in flashcards:
                database.create_flashcard(self.current_note_id, card["question"], card["answer"])
            result = self._format_qa_list(flashcards, f"{len(flashcards)} flashcards created:")
        except Exception as e:
            result = f"Error: {e}"
        self.after(0, self._generate_flashcards_done, result)

    def _generate_flashcards_done(self, result):
        self._show_ai_output(result)
        self.flashcards_button.configure(state="normal", text="Generate Flashcards")

    # ---- Suggest Schedule -------------------------------------------------

    def run_suggest_schedule(self):
        self.schedule_button.configure(state="disabled", text="Thinking...")
        threading.Thread(target=self._suggest_schedule_worker, daemon=True).start()

    def _suggest_schedule_worker(self):
        try:
            due_items = database.get_due_flashcards()
            result = ai.suggest_schedule(due_items)
        except Exception as e:
            result = f"Error: {e}"
        self.after(0, self._suggest_schedule_done, result)

    def _suggest_schedule_done(self, result):
        self._show_ai_output(result)
        self.schedule_button.configure(state="normal", text="Suggest Schedule")

    # ---- Chat (text + voice) -----------------------------------------------

    def _append_chat(self, text):
        """Append to the read-only chat display without clearing history."""
        text = clean_markdown_artifacts(text)
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text)
        self.chat_display.configure(state="disabled")

    def send_chat_message(self):
        message = self.chat_entry.get()
        if not message:
            return

        self._append_chat(f"You: {message}\n\n")
        self.chat_entry.delete(0, "end")

        self.chat_send_button.configure(state="disabled", text="Thinking...")
        self.chat_record_button.configure(state="disabled")
        threading.Thread(target=self._send_chat_worker, args=(message,), daemon=True).start()

    def _send_chat_worker(self, message):
        try:
            result = ai.chat(message)
        except Exception as e:
            result = f"Error: {e}"
        self.after(0, self._send_chat_done, result)

    def _send_chat_done(self, result):
        self._append_chat(f"AI: {result}\n\n")
        self.chat_send_button.configure(state="normal", text="Send")
        self.chat_record_button.configure(state="normal")

    def record_voice_message(self):
        self.chat_send_button.configure(state="disabled")
        self.chat_record_button.configure(state="disabled", text="Recording...")
        threading.Thread(target=self._record_voice_worker, daemon=True).start()

    def _record_voice_worker(self):
        try:
            path = _record_audio()
            text = ai.transcribe(path)
            os.remove(path)
            error = None
        except Exception as e:
            text = None
            error = f"Error: {e}"
        self.after(0, self._record_voice_done, text, error)

    def _record_voice_done(self, text, error):
        if error is not None:
            self._append_chat(f"{error}\n\n")
        else:
            self.chat_entry.delete(0, "end")
            self.chat_entry.insert(0, text)
        self.chat_send_button.configure(state="normal")
        self.chat_record_button.configure(state="normal", text="Record")

    # ---- Flashcards review (SM-2) ------------------------------------------

    def _set_flashcard_display(self, text):
        """Write to the read-only flashcard display box."""
        text = clean_markdown_artifacts(text)
        self.flashcard_display.configure(state="normal")
        self.flashcard_display.delete("1.0", "end")
        self.flashcard_display.insert("1.0", text)
        self.flashcard_display.configure(state="disabled")

    def _set_rating_buttons_state(self, state):
        self.again_button.configure(state=state)
        self.hard_button.configure(state=state)
        self.good_button.configure(state=state)
        self.easy_button.configure(state=state)

    def load_due_flashcards(self):
        """Reload the due-flashcards queue and show the first card."""
        self.due_flashcards = database.get_due_flashcards()
        self.current_card_index = 0
        self.show_current_card()

    def show_current_card(self):
        """Show the current card's question, or a no-cards-due message."""
        if not self.due_flashcards or self.current_card_index >= len(self.due_flashcards):
            self._set_flashcard_display("No cards due right now!")
            self.flashcard_progress.configure(text="No cards due")
            self.show_answer_button.configure(state="disabled")
            self._set_rating_buttons_state("disabled")
            return

        card = self.due_flashcards[self.current_card_index]
        self._set_flashcard_display(card["question"])
        self.answer_visible = False
        self.flashcard_progress.configure(
            text=f"Card {self.current_card_index + 1} of {len(self.due_flashcards)}"
        )
        self.show_answer_button.configure(state="normal")
        self._set_rating_buttons_state("disabled")

    def show_answer(self):
        """Reveal the current card's answer underneath the question."""
        card = self.due_flashcards[self.current_card_index]
        self._set_flashcard_display(f"{card['question']}\n\n---\n\n{card['answer']}")
        self.answer_visible = True
        self.show_answer_button.configure(state="disabled")
        self._set_rating_buttons_state("normal")

    def rate_card(self, quality):
        """Apply an SM-2 review for the current card, then advance."""
        card = self.due_flashcards[self.current_card_index]
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
        self.current_card_index += 1
        self.show_current_card()

    # ---- Quiz (AI-generated, self-graded) ----------------------------------

    def _set_quiz_display(self, text):
        """Write to the read-only quiz question/answer box."""
        text = clean_markdown_artifacts(text)
        self.quiz_question_display.configure(state="normal")
        self.quiz_question_display.delete("1.0", "end")
        self.quiz_question_display.insert("1.0", text)
        self.quiz_question_display.configure(state="disabled")

    def run_generate_quiz(self):
        if self.current_note_id is None:
            self._set_quiz_display("Save this note before generating a quiz.")
            return
        content = self.content_textbox.get("1.0", "end-1c")
        self.generate_quiz_button.configure(state="disabled", text="Generating...")
        threading.Thread(target=self._generate_quiz_worker, args=(content,), daemon=True).start()

    def _generate_quiz_worker(self, content):
        try:
            quiz = ai.generate_quiz(content)
            self.current_quiz = quiz
            self.quiz_note_id = self.current_note_id
            self.quiz_index = 0
            self.quiz_score = {"correct": 0, "total": 0}
            error = None
        except Exception as e:
            error = f"Error: {e}"
        self.after(0, self._generate_quiz_done, error)

    def _generate_quiz_done(self, error):
        self.generate_quiz_button.configure(state="normal", text="Generate Quiz")
        if error is not None:
            self._set_quiz_display(error)
            return
        self.show_current_quiz_question()

    def show_current_quiz_question(self):
        """Show the current quiz question, or a completion summary."""
        if self.quiz_index >= len(self.current_quiz):
            correct = self.quiz_score["correct"]
            total = self.quiz_score["total"]
            if total:
                self._set_quiz_display(f"Quiz complete! {correct}/{total} correct")
            else:
                self._set_quiz_display("Generate a quiz to get started.")
            self.quiz_progress.configure(text=f"{correct}/{total} correct")

            self.quiz_answer_entry.configure(state="normal")
            self.quiz_answer_entry.delete(0, "end")
            self.quiz_answer_entry.configure(state="disabled")
            self.quiz_submit_button.configure(state="disabled")
            self.quiz_correct_button.configure(state="disabled")
            self.quiz_incorrect_button.configure(state="disabled")
            return

        question = self.current_quiz[self.quiz_index]
        self._set_quiz_display(question["question"])
        self.quiz_progress.configure(
            text=(
                f"Question {self.quiz_index + 1} of {len(self.current_quiz)} — "
                f"{self.quiz_score['correct']}/{self.quiz_score['total']} correct so far"
            )
        )

        self.quiz_answer_entry.configure(state="normal")
        self.quiz_answer_entry.delete(0, "end")
        self.quiz_submit_button.configure(state="normal")
        self.quiz_correct_button.configure(state="disabled")
        self.quiz_incorrect_button.configure(state="disabled")

    def submit_quiz_answer(self):
        """Reveal the correct answer next to the user's typed answer."""
        question = self.current_quiz[self.quiz_index]
        self.quiz_submitted_answer = self.quiz_answer_entry.get()

        self._set_quiz_display(
            f"Q: {question['question']}\n\n"
            f"Your answer: {self.quiz_submitted_answer}\n"
            f"Correct answer: {question['answer']}"
        )

        self.quiz_answer_entry.configure(state="disabled")
        self.quiz_submit_button.configure(state="disabled")
        self.quiz_correct_button.configure(state="normal")
        self.quiz_incorrect_button.configure(state="normal")

    def grade_quiz_answer(self, is_correct: bool):
        """Log the self-graded attempt, update the score, advance."""
        question = self.current_quiz[self.quiz_index]
        database.log_quiz_attempt(
            self.quiz_note_id,
            question["question"],
            question["answer"],
            self.quiz_submitted_answer,
            is_correct,
        )

        self.quiz_score["total"] += 1
        if is_correct:
            self.quiz_score["correct"] += 1

        self.quiz_index += 1
        self.show_current_quiz_question()

    # ---- Progress (read-only stats) ----------------------------------------

    def load_progress_stats(self):
        """Refresh every section of the Progress tab from the database."""
        stats = database.get_quiz_stats()

        overall = stats["overall"]
        if overall["total"]:
            self.progress_overall_label.configure(
                text=f"Overall: {overall['correct']}/{overall['total']} correct ({overall['percent']}%)"
            )
        else:
            self.progress_overall_label.configure(text="No quiz attempts yet.")

        total_flashcards = len(database.get_all_flashcards())
        due_flashcards = len(database.get_due_flashcards())
        self.progress_flashcards_label.configure(
            text=f"{total_flashcards} flashcards total, {due_flashcards} due now"
        )

        for widget in self.progress_by_note_frame.winfo_children():
            widget.destroy()
        for entry in stats["by_note"]:
            label_text = (
                f"{entry['subject']} — {entry['title']}: "
                f"{entry['correct']}/{entry['total']} correct ({entry['percent']}%)"
            )
            ctk.CTkLabel(self.progress_by_note_frame, text=label_text).pack(anchor="w", padx=5, pady=2)

        for widget in self.progress_recent_frame.winfo_children():
            widget.destroy()
        for attempt in database.get_quiz_history()[:10]:
            mark = "✓" if attempt["is_correct"] else "✗"
            label_text = (
                f"{mark} {attempt['question']} — you said: {attempt['user_answer']} "
                f"(correct: {attempt['correct_answer']})"
            )
            ctk.CTkLabel(self.progress_recent_frame, text=label_text).pack(anchor="w", padx=5, pady=2)


def main():
    database.init_db()
    app = NoteForgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
