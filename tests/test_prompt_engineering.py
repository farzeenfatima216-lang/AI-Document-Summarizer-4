import unittest

from prompt_engineering import (
    RECOMMENDED_VARIANT,
    build_prompt,
    evaluate_prompt_variants,
)


class PromptEngineeringTests(unittest.TestCase):
    def test_recommended_prompts_are_grounded(self):
        self.assertEqual(RECOMMENDED_VARIANT, "structured_grounded")
        self.assertIn("only", build_prompt("qa", context="facts", question="What?"))
        self.assertIn("Not found", build_prompt("qa", context="facts", question="What?"))

    def test_evaluation_compares_quality_time_and_tokens(self):
        cases = [{"document": "The report has 3 findings."}]
        results = evaluate_prompt_variants(
            "summary",
            cases,
            invoke=lambda prompt: prompt,
            score=lambda case, response: 1.0 if case["document"] in response else 0.0,
        )

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result["quality"], 1.0)
            self.assertGreater(result["estimated_input_tokens"], 0)
            self.assertGreaterEqual(result["response_time_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
