import unittest
from unittest.mock import patch

from src import state


class SessionStateTests(unittest.TestCase):
    def test_mutable_defaults_are_not_shared_between_sessions(self):
        first_session = {}
        with patch.object(state.st, "session_state", first_session):
            state.init_state()
        first_session["log"].append({"score": 42})
        first_session["user_profile"]["name"] = "First archer"

        second_session = {}
        with patch.object(state.st, "session_state", second_session):
            state.init_state()

        self.assertEqual(second_session["log"], [])
        self.assertEqual(second_session["user_profile"]["name"], "")
        self.assertEqual(second_session["coach_mode"], "book")

    def test_old_rules_session_migrates_to_book_once(self):
        old_session = {"coach_mode": "rules"}
        with patch.object(state.st, "session_state", old_session):
            state.init_state()

        self.assertEqual(old_session["coach_mode"], "book")
        self.assertEqual(old_session["coach_version"], 2)

    def test_explicit_rules_choice_persists_after_migration(self):
        current_session = {"coach_mode": "rules", "coach_version": 2}
        with patch.object(state.st, "session_state", current_session):
            state.init_state()

        self.assertEqual(current_session["coach_mode"], "rules")


if __name__ == "__main__":
    unittest.main()
