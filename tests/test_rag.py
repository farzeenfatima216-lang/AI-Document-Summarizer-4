import unittest

from rag import FAISSRetriever

try:
    import faiss  # noqa: F401
    import numpy  # noqa: F401
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vocabulary = ("python", "database", "quantum")
        vectors = []
        for text in texts:
            lower_text = text.lower()
            vectors.append([lower_text.count(term) for term in vocabulary])
        return vectors


@unittest.skipUnless(
    FAISS_AVAILABLE,
    "faiss-cpu and numpy are required for retrieval tests. "
    "Install with: pip install faiss-cpu numpy",
)
class RAGTests(unittest.TestCase):
    def test_retrieves_semantically_matching_chunk(self):
        retriever = FAISSRetriever(
            [
                "Python is used for data processing.",
                "A database stores records and indexes.",
                "Quantum computing uses qubits.",
            ],
            embedding_model=FakeEmbeddingModel(),
        )

        results = retriever.retrieve(
            "How does a database store records?", top_k=1
        )

        self.assertEqual(results[0]["index"], 1)
        self.assertIn("database", results[0]["chunk"])

    def test_top_k_limits_context(self):
        retriever = FAISSRetriever(
            ["Python Python", "database", "quantum"],
            embedding_model=FakeEmbeddingModel(),
        )

        self.assertEqual(len(retriever.retrieve("Python", top_k=2)), 2)

    def test_returns_source_citation_metadata(self):
        retriever = FAISSRetriever(
            [{
                "text": "Database records are indexed.",
                "page_number": 4,
                "paragraph_number": 2,
                "section": "Storage",
            }],
            embedding_model=FakeEmbeddingModel(),
        )

        result = retriever.retrieve("database records", top_k=1)[0]

        self.assertEqual(result["page_number"], 4)
        self.assertEqual(result["paragraph_number"], 2)
        self.assertEqual(result["section"], "Storage")


if __name__ == "__main__":
    unittest.main()
