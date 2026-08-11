import unittest

from serve_frontend import build_answer, fallback_sources, parse_form_data


class BackendUploadTests(unittest.TestCase):
    def test_fallback_does_not_return_unrelated_first_chunk(self):
        answer = build_answer(
            "How can young people join civic activities?",
            "Name: Farzeen Fatima",
            [{"text": "Name: Farzeen Fatima", "paragraph_number": 1}],
        )

        self.assertEqual(answer, "Not found in the document.")

    def test_fallback_prefers_definition_over_matching_heading(self):
        answer = build_answer(
            "what is AI",
            "What is Artificial Intelligence (AI)?\n"
            "Artificial Intelligence (AI) is a branch of computer science "
            "that focuses on designing intelligent systems.",
            [
                {"text": "What is Artificial Intelligence (AI)?"},
                {
                    "text": "Artificial Intelligence (AI) is a branch of "
                    "computer science that focuses on designing intelligent systems."
                },
            ],
        )

        self.assertIn("branch of computer science", answer)

    def test_parse_form_data_reads_multipart_file_upload(self):
        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="sample.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            "Hello from upload test\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        form = parse_form_data(body, f"multipart/form-data; boundary={boundary}")
        file_item = form["file"]

        self.assertEqual(file_item.filename, "sample.txt")
        self.assertEqual(file_item.file.read(), b"Hello from upload test")

    def test_fallback_sources_are_displayable_retrieval_records(self):
        sources = fallback_sources(
            [{"text": "IEEE develops standards.", "page_number": 2}],
            "What does IEEE develop?",
        )

        self.assertEqual(sources[0]["chunk"], "IEEE develops standards.")
        self.assertEqual(sources[0]["page_number"], 2)
        self.assertGreater(sources[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
