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


def advice_for(
    metrics_value,
    shape,
    avg=7.5,
    log=None,
    bow="recurve",
    mode="book",
    quality=None,
    handedness="right",
    self_report="none",
    session_context=None,
):
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
        self_report=self_report,
        session_context=session_context,
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

    def test_single_escaped_arrow_does_not_trigger_a_whole_group_rebuild(self):
        value = metrics(spread=31, sx=35, sy=18)
        value.update({
            "n": 6,
            "anisotropy": 1.94,
            "outlier": {
                "present": True,
                "index": 5,
                "distance": 150.0,
                "core_spread": 8.0,
                "improvement_ratio": 0.74,
            },
        })

        result = advice_for(value, "horizontal")

        self.assertEqual(result["diagnosis"]["key"], "single_outlier")
        self.assertEqual(result["book_source"]["id"], "feel_then_check")
        self.assertIn("第 6 箭", result["diagnosis"]["evidence"])
        self.assertIn("不要因为一支异常箭", result["feedback"]["do_not_change"])

    def test_coaching_uses_the_detected_five_arrow_end_size(self):
        value = metrics(spread=31, sx=35, sy=18)
        value.update({
            "n": 5,
            "anisotropy": 1.94,
            "outlier": {
                "present": True,
                "index": 4,
                "distance": 150.0,
                "core_spread": 8.0,
                "improvement_ratio": 0.74,
            },
        })

        result = advice_for(value, "horizontal")

        self.assertIn("4/5", result["feedback"]["success_criterion"])
        self.assertIn("5箭", result["drill"]["name"])
        self.assertNotIn("6箭", result["drill"]["name"])

    def test_feedback_exposes_uncertainty_and_a_measurable_next_end_test(self):
        result = advice_for(metrics(spread=42, sx=40, sy=9), "horizontal")

        self.assertEqual(result["diagnosis"]["confidence"], "medium")
        self.assertEqual(len(result["feedback"]["alternative_hypotheses"]), 3)
        self.assertIn("15%", result["feedback"]["success_criterion"])
        self.assertIn("0.5", result["feedback"]["success_criterion"])

    def test_archer_observation_selects_a_matching_book_experiment(self):
        result = advice_for(
            metrics(spread=42, sx=40, sy=9),
            "horizontal",
            self_report="anchor_unclear",
        )

        self.assertEqual(result["book_source"]["id"], "anchor_balance")
        self.assertEqual(result["feedback"]["selected_by"], "self_report")
        self.assertEqual(result["diagnosis"]["self_report_label"], "锚点不清楚")

    def test_history_comparison_ignores_a_different_distance(self):
        previous_metrics = metrics(spread=60, sx=55, sy=12)
        previous = advice_for(previous_metrics, "horizontal")
        log = [{
            "distance_m": 18,
            "target_face": "40cm_10ring",
            "metrics": previous_metrics,
            "advice": previous,
        }]

        result = advice_for(
            metrics(spread=42, sx=40, sy=9),
            "horizontal",
            log=log,
            session_context={"distance_m": 30, "target_face": "80cm_10ring"},
        )

        self.assertEqual(result["diagnosis"]["trend_key"], "first")
        self.assertFalse(result["rag"]["history_used"])

    def test_repeated_pattern_in_the_same_conditions_raises_confidence(self):
        current = metrics(spread=42, sx=40, sy=9)
        previous = advice_for(current, "horizontal")
        log = [{
            "distance_m": 30,
            "target_face": "80cm_10ring",
            "metrics": current,
            "advice": previous,
        }]

        result = advice_for(
            current,
            "horizontal",
            log=log,
            session_context={"distance_m": 30, "target_face": "80cm_10ring"},
        )

        self.assertEqual(result["diagnosis"]["confidence"], "high")

    def test_an_improving_experiment_is_repeated_and_marked_effective(self):
        previous_metrics = metrics(spread=50, sx=46, sy=10)
        previous = advice_for(previous_metrics, "horizontal")
        log = [{
            "distance_m": 30,
            "target_face": "80cm_10ring",
            "metrics": previous_metrics,
            "advice": previous,
        }]

        result = advice_for(
            metrics(spread=42, sx=39, sy=9),
            "horizontal",
            log=log,
            session_context={"distance_m": 30, "target_face": "80cm_10ring"},
        )

        self.assertEqual(result["diagnosis"]["trend_key"], "improving")
        self.assertEqual(result["book_source"]["id"], previous["book_source"]["id"])
        self.assertEqual(result["feedback"]["previous_experiment"]["verdict"], "暂定有效")


if __name__ == "__main__":
    unittest.main()
