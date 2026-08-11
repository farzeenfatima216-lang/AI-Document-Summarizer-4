import unittest
import zipfile
from io import BytesIO

import utils
from utils import extract_text_from_file, prepare_document_payload


class DocumentProcessingTests(unittest.TestCase):
    def test_extracts_text_from_docx_without_external_dependencies(self):
        xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
  <w:body>
    <w:p>
      <w:r><w:t>Sample DOCX content</w:t></w:r>
    </w:p>
  </w:body>
</w:document>"""

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml_payload)

        text = extract_text_from_file("sample.docx", buffer.getvalue())

        self.assertIn("Sample DOCX content", text)

    def test_extracts_text_from_plain_text_files(self):
        text = extract_text_from_file("notes.txt", b"Plain text upload works")
        self.assertEqual(text, "Plain text upload works")

    def test_prepares_source_metadata_for_text_chunks(self):
        payload = prepare_document_payload(
            "notes.txt",
            b"Overview\n\nThis is the first paragraph.\n\nDetails:\nMore detail.",
        )

        self.assertEqual(payload["chunk_records"][0]["paragraph_number"], 1)
        self.assertEqual(payload["chunk_records"][0]["section"], "Overview")
        self.assertIsNone(payload["chunk_records"][0]["page_number"])

    def test_detects_scanned_pdf_and_uses_ocr(self):
        original_reader = utils.PdfReader
        original_fitz = utils.fitz
        original_ocr_reader = utils.OCR_READER
        original_image = utils.Image

        class EmptyPage:
            def extract_text(self):
                return ""

        class FakeReader:
            def __init__(self, stream):
                self.pages = [EmptyPage()]

        class FakePixmap:
            def tobytes(self, format_name):
                return b"image"

        class FakePage:
            def get_pixmap(self, matrix, alpha):
                return FakePixmap()

            def get_text(self):
                return ""

        class FakeDocument:
            def __iter__(self):
                return iter([FakePage()])

            def close(self):
                pass

        class FakeFitz:
            Matrix = staticmethod(lambda x, y: (x, y))

            @staticmethod
            def open(stream, filetype):
                return FakeDocument()

        class FakeOCRReader:
            def readtext(self, image, detail=0, paragraph=True):
                return ["OCR extracted content"]

        class FakeImageObject:
            mode = "L"
            width = 100
            height = 100

            def convert(self, mode):
                return self

            def filter(self, filt):
                return self

            def resize(self, size, resample):
                return self

            def point(self, fn, mode):
                return self

        class FakeImage:
            @staticmethod
            def open(stream):
                return FakeImageObject()

        original_imageops = utils.ImageOps
        original_imagefilter = utils.ImageFilter

        try:
            utils.PdfReader = FakeReader
            utils.fitz = FakeFitz
            utils.OCR_READER = FakeOCRReader()
            utils.Image = FakeImage
            utils.ImageOps = None
            utils.ImageFilter = None
            payload = prepare_document_payload("scan.pdf", b"pdf bytes")
        finally:
            utils.PdfReader = original_reader
            utils.fitz = original_fitz
            utils.OCR_READER = original_ocr_reader
            utils.Image = original_image
            utils.ImageOps = original_imageops
            utils.ImageFilter = original_imagefilter

        self.assertEqual(payload["text"], "OCR extracted content")
        self.assertEqual(payload["extraction_method"], "ocr")
        self.assertTrue(payload["scanned_pdf"])
        self.assertEqual(payload["ocr_pages"], [1])


if __name__ == "__main__":
    unittest.main()
