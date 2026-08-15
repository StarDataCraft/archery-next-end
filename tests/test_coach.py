import unittest

from src.coach import CoachConfig, CoachRAG


BASE = {
    "title": "base",
    "single_cue": "base cue",
    "pass_fail": "base pass",
    "fallback": "base fallback",
    "drill": {"name": "base drill", "how": "base", "duration_s": 30},
}


def metrics(*, spread, sx, sy, dx=0.0, dy=0.0):
    mag = (dx * dx + dy * dy) ** 0.5
    return {
        "spread": spread,
        "spread_ratio": spread / 405.0,
        "sx": sx,
        "sy": sy,
        "slope_deg": 0.0,
        "offset": {"dx": dx, "dy": dy, "mag": mag},
        "offset_ratio": mag / 405.0,
    }


def advice_for(metrics_value, shape, avg=7.5, log=None, bow="recurve", mode="book", quality=None, handedness="right"):
    coach = CoachRAG(CoachConfig(mode=mode))
    return coach.enhance_advice(
        base_advice=BASE,
        metrics=metrics_value,
        shape=shape,
        handedness=handedness,
        lang="zh",
        scoring={"avg": avg, "total": avg * 6},
        user_profile={"bow": bow, "goals": "", "recurring_issues": "", "constraints": ""},
        log=log or [],
        quality=quality,
    )


class BookCoachTests(unittest.TestCase):
    def test_distinct_group_patterns_produce_distinct_book_advice(self):
        horizontal = advice_for(metrics(spread=42, sx=40, sy=9), "horizontal")
        vertical = advice_for(metrics(spread=42, sx=9, sy=40), "vertical")
        loose = advice_for(metrics(spread=80, sx=55, sy=52), "round", avg=5.0)
        offset = advice_for(metrics(spread=16, sx=12, sy=11, dx=85, dy=-35), "round", avg=8.5)
        strong = advice_for(metrics(spread=15, sx=11, sy=10), "round", avg=9.5)

        ids = {
            horizontal["book_source"]["id"],
            vertical["book_source"]["id"],
            loose["book_source"]["id"],
            offset["book_source"]["id"],
            strong["book_source"]["id"],
        }

        self.assertEqual(horizontal["diagnosis"]["key"], "horizontal")
        self.assertEqual(vertical["diagnosis"]["key"], "vertical")
        self.assertEqual(loose["diagnosis"]["key"], "loose")
        self.assertEqual(offset["diagnosis"]["key"], "tight_offset")
        self.assertEqual(strong["diagnosis"]["key"], "protect")
        self.assertEqual(len(ids), 5)
        self.assertNotEqual(horizontal["single_cue"], vertical["single_cue"])

    def test_saved_history_rotates_within_the_same_relevant_issue(self):
        current_metrics = metrics(spread=42, sx=40, sy=9)
        first = advice_for(current_metrics, "horizontal")
        log = [{"metrics": current_metrics, "advice": first}]

        second = advice_for(current_metrics, "horizontal", log=log)

        self.assertEqual(second["diagnosis"]["key"], "horizontal")
        self.assertNotEqual(first["book_source"]["id"], second["book_source"]["id"])
        self.assertEqual(second["diagnosis"]["trend_key"], "steady")

    def test_repeated_issue_becomes_a_recorded_experiment(self):
        current_metrics = metrics(spread=42, sx=40, sy=9)
        log = [
            {
                "metrics": current_metrics,
                "advice": {"diagnosis": {"key": "horizontal"}, "book_source": {"id": "string_vertical"}},
            },
            {
                "metrics": current_metrics,
                "advice": {"diagnosis": {"key": "horizontal"}, "book_source": {"id": "bow_hand_relaxed"}},
            },
        ]

        result = advice_for(current_metrics, "horizontal", log=log)

        self.assertEqual(result["book_source"]["id"], "diary_experiment")

    def test_tight_offset_for_barebow_does_not_suggest_moving_a_sight(self):
        result = advice_for(metrics(spread=14, sx=10, sy=9, dx=80), "round", bow="barebow")

        self.assertEqual(result["diagnosis"]["key"], "tight_offset")
        self.assertEqual(result["book_source"]["id"], "anchor_balance")
        self.assertNotIn("瞄具", result["single_cue"])

    def test_low_image_quality_marks_the_diagnosis_as_a_hypothesis(self):
        result = advice_for(
            metrics(spread=42, sx=40, sy=9),
            "horizontal",
            quality={"score": 0.3, "flags": ["weak_circle"]},
        )

        self.assertEqual(result["diagnosis"]["confidence"], "low")
        self.assertIn("待验证假设", result["diagnosis"]["evidence"])

    def test_legacy_rag_mode_uses_the_reliable_book_engine(self):
        result = advice_for(metrics(spread=42, sx=40, sy=9), "horizontal", mode="rag")

        self.assertEqual(result["rag"]["engine"], "reviewed_book_cards")
        self.assertIn("pdf_pages", result["book_source"])

    def test_handedness_is_reflected_in_the_coaching_context(self):
        result = advice_for(metrics(spread=42, sx=40, sy=9), "horizontal", handedness="left")

        self.assertIn("右手持弓，左手拉弦", result["diagnosis"]["handedness_context"])


if __name__ == "__main__":
    unittest.main()
