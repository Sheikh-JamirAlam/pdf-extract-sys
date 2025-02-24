# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
import io
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import cv2

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected request body model
class PDFRequest(BaseModel):
    pdf_url: str

def is_searchable(page):
    """
    Check if the page is searchable.
    If page.get_text() returns any text, we consider it searchable.
    """
    text = page.get_text().strip()
    return bool(text)

@app.post("/extract")
async def extract_pdf(pdf_req: PDFRequest):
    # Download the PDF using an asynchronous HTTP client
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_req.pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    pdf_data = response.content

    # Validate file size (limit: 50MB)
    if len(pdf_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF file too large")

    # Open the PDF using PyMuPDF
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    if doc.page_count > 2000:
        raise HTTPException(status_code=400, detail="PDF has too many pages")

    results = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        if is_searchable(page):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        spans = line["spans"]
                        current_sentence = ""
                        current_words = []
                        current_span = None
                        
                        for span in spans:
                            words = span["text"].split()
                            span_width = span["bbox"][2] - span["bbox"][0]
                            char_width = span_width / len(span["text"]) if span["text"] else 0
                            
                            word_start = span["bbox"][0]
                            for word in words:
                                word_width = len(word) * char_width
                                word_bbox = [
                                    word_start,
                                    span["bbox"][1],
                                    word_start + word_width,
                                    span["bbox"][3]
                                ]
                                current_words.append((word, word_bbox, span))
                                word_start += (len(word) + 1) * char_width  # +1 for space
                                current_sentence += word + " "
                                
                                if any(word.endswith(end) for end in ['.', '!', '?']):
                                    if current_sentence.strip():
                                        # Calculate bbox from first to last word of sentence
                                        first_word = current_words[0]
                                        last_word = current_words[-1]
                                        bbox = [
                                            (first_word[1][0] / page.rect.width) * 100,
                                            (first_word[1][1] / page.rect.height) * 100,
                                            (last_word[1][2] / page.rect.width) * 100,
                                            (last_word[1][3] / page.rect.height) * 100
                                        ]
                                        
                                        results.append({
                                            "text": current_sentence.strip(),
                                            "bbox": bbox,
                                            "pageNumber": page_num
                                        })
                                        
                                        # Start next sentence with remaining words
                                        current_sentence = ""
                                        current_words = []
                        
                        # Handle remaining text in the line
                        if current_sentence.strip():
                            first_word = current_words[0]
                            last_word = current_words[-1]
                            bbox = [
                                (first_word[1][0] / page.rect.width) * 100,
                                (first_word[1][1] / page.rect.height) * 100,
                                (last_word[1][2] / page.rect.width) * 100,
                                (last_word[1][3] / page.rect.height) * 100
                            ]
                            
                            results.append({
                                "text": current_sentence.strip(),
                                "bbox": bbox,
                                "pageNumber": page_num
                            })
        else:
            # OCR part
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            ocr_text = pytesseract.image_to_string(
                img_data, 
                config='--psm 6'
            )
            lines = ocr_text.splitlines()
            
            for line in lines:
                if line.strip():
                    results.append({
                        "text": line.strip(),
                        "bbox": [0, 0, 0, 0],
                        "pageNumber": page_num
                    })

    return {"results": results}
