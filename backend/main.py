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
    current_line = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        if is_searchable(page):
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        line_text = " ".join(span["text"] for span in line["spans"]).strip()
                        if line_text:
                            first_span = line["spans"][0]
                            last_span = line["spans"][-1]
                            # Convert coordinates to percentages
                            bbox = [
                                (first_span["bbox"][0] / page_width) * 100,  # left
                                (first_span["bbox"][1] / page_height) * 100, # top
                                (last_span["bbox"][2] / page_width) * 100,   # right
                                (last_span["bbox"][3] / page_height) * 100   # bottom
                            ]
                            results.append({
                                "text": line_text,
                                "bbox": bbox,
                                "pageNumber": page_num
                            })
        else:
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
