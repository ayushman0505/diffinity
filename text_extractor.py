# utils/text_extractor.py

import pdfplumber
import pytesseract
from PIL import Image
import tempfile
import os


def extract_text_from_pdf(pdf_file):
    """
    Extracts text line-by-line from a PDF file.
    """
    text_blocks = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                text_blocks.extend([line.strip() for line in lines if line.strip()])
    return text_blocks


def extract_text_from_image(image_file):
    """
    Extracts text from an image file using pytesseract OCR.
    """
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def handle_uploaded_file(uploaded_file):
    """
    Saves uploaded file to a temporary location and returns the path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def cleanup_temp_file(file_path):
    """
    Deletes a temporary file after use.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
