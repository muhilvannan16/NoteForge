from datetime import date, timedelta

from core.srs import ReviewState, review


def test_fresh_card_perfect_recall():
    result = review(ReviewState(), 5)
    assert result.ease_factor == 2.6
    assert result.interval_days == 1
    assert result.repetitions == 1
    assert result.due_date == date.today() + timedelta(days=1)


def test_fresh_card_failure():
    result = review(ReviewState(), 2)
    assert abs(result.ease_factor - 2.18) < 0.001
    assert result.interval_days == 1
    assert result.repetitions == 0


def test_five_perfect_reviews_in_a_row():
    state = ReviewState()
    expected_ease = [2.6, 2.7, 2.8, 2.9, 3.0]
    expected_interval = [1, 6, 16, 45, 131]
    expected_reps = [1, 2, 3, 4, 5]
    for i in range(5):
        state = review(state, 5)
        assert abs(state.ease_factor - expected_ease[i]) < 0.001
        assert state.interval_days == expected_interval[i]
        assert state.repetitions == expected_reps[i]


def test_failure_resets_repetitions_mid_streak():
    state = ReviewState()
    qualities = [5, 5, 1, 5]
    expected_ease = [2.6, 2.7, 2.16, 2.26]
    expected_interval = [1, 6, 1, 1]
    expected_reps = [1, 2, 0, 1]
    for i, q in enumerate(qualities):
        state = review(state, q)
        assert abs(state.ease_factor - expected_ease[i]) < 0.001
        assert state.interval_days == expected_interval[i]
        assert state.repetitions == expected_reps[i]
