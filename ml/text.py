"""Various text-extraction and related utilities"""

from __future__ import annotations

import mimetypes

from subprocess import check_output

import pytesseract

def get_pdf_text(path: str, *args) -> str:
    """Returns raw text from a pdf.

    This calls the `pdftotext` command-line utility to extract text from the pdf.
    We call it with the path to the pdf and the `-` argument to write to stdout.
    You can pass additional arguments to `pdftotext` as additional arguments to this function.
    """
    args = ["pdftotext", path, "-", ]
    out = check_output(args).decode("utf-8", "replace")
    return out

def get_ocr_text(path: str, **kw) -> str:
    """Gets text from an image using OCR.

    This calls the python bindings to `pytesseract` to extract text from the image.
    You can pass additional keyword arguments to `pytesseract.image_to_string` as additional
    keyword arguments to this function.
    """
    out = pytesseract.image_to_string(path, **kw)
    return out

def get_text(path: str, *args, **kw) -> str:
    """Returns the text from the given file.

    This uses the file extension to determine how to extract the text.
    If it's a pdf, we use `pdftotext` to extract the text (with additional *args).
    If it's an image, we use OCR to extract the text (with additional **kw).
    Else we assume it's a text file, and just read the text directly.
    """
    type, enc = mimetypes.guess_type(path)
    if enc == '.pdf':
        out = get_pdf_text(path, *args)
    elif type.startswith('image'):
        out = get_ocr_text(path, **kw)
    else:
        with open(path) as f:
            out = f.read()
    return out
