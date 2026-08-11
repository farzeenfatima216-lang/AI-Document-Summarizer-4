import unittest

from document_evaluation import compare_pdf_extraction


class DocumentEvaluationTests(unittest.TestCase):
    def test_compares_normal_and_ocr_accuracy(self):
        result = compare_pdf_extraction(
            "The report contains three findings.",
            "The report contains three findings.",
            "The report contains findings.",
        )

        self.assertEqual(result["normal_pdf"]["f1"], 1.0)
        self.assertLess(result["ocr"]["recall"], result["normal_pdf"]["recall"])


if __name__ == "__main__":
    unittest.main()
