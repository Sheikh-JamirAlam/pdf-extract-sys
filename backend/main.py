from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import io
import pypdfium2 as pdfium
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PDFRequest(BaseModel):
    pdf_url: str

@app.post("/extract")
async def extract_pdf(pdf_req: PDFRequest):
    # Download the PDF
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_req.pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    
    pdf_data = io.BytesIO(response.content)  # Convert bytes to a file-like object
    results = []
    sentence_enders = {".", "!", "?"}  # Sentence-ending punctuation
    email_domains = {"com", "org", "edu"}  # Email top level domains
    
    # Define patterns for various content types
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    equation_pattern = re.compile(r'[OΟ]\([^)]+\)|(\([^)]+\))')
    decimal_pattern = re.compile(r'\b\d+\.\d+\b')
    toc_pattern = re.compile(r'\b\d+\.\d+.*?\.{2,}.*?\d+\b')

    # Open the PDF using PDFium
    pdf = pdfium.PdfDocument(pdf_data)
    
    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)
        textpage = page.get_textpage()
        page_width, page_height = page.get_size()

        current_sentence = ""
        sentence_chars = []
        full_text = textpage.get_text_range()

        # Iterate through individual characters and their bounding boxes
        for index, char in enumerate(full_text):
            try:
                bbox = textpage.get_charbox(index, loose=False)
            except IndexError:
                continue  # Skip if index is out of range

            if not bbox:
                continue  # Skip if there's no bounding box (e.g., spaces)
                
            # Normalize bounding box
            bbox_normalized = [
                (bbox[0] / page_width) * 100,
                ((page_height - bbox[3]) / page_height) * 100,
                (bbox[2] / page_width) * 100,
                ((page_height - bbox[1]) / page_height) * 100
            ]

            sentence_chars.append((char, bbox_normalized))
            current_sentence += char
            
            # Check for decimals
            if char == "." and len(current_sentence) >= 2 and current_sentence[-2].isdigit() and full_text[index + 1].isdigit():
                continue
            # Check for mathematical sequences
            if char == "." and (full_text[index + 1] == "." or full_text[index - 1] == "."):
                continue
            # Check for emails
            if char == "." and (full_text[index + 1 : index + 4] in email_domains):
                continue
                    
            # If a sentence ender or line break is found, finalize sentence
            if char in sentence_enders or char == "\n" or char == "\ufffe":
                if current_sentence.strip():
                    
                    while sentence_chars and sentence_chars[0][0] in {"\r", "\n"}:
                        sentence_chars.pop(0)
                    while sentence_chars and sentence_chars[-1][0] in {"\r", "\n"}:
                        sentence_chars.pop()
                    
                    sentence_bbox = [
                        min(b[0] for _, b in sentence_chars),  # x0
                        min(b[1] for _, b in sentence_chars),  # y0
                        max(b[2] for _, b in sentence_chars),  # x1
                        max(b[3] for _, b in sentence_chars)   # y1
                    ]

                    results.append({
                        "text": current_sentence.strip(),
                        "bbox": sentence_bbox,
                        "pageNumber": page_num
                    })

                    # Reset sentence tracking
                    current_sentence = ""
                    sentence_chars = []

        # Handle remaining text if no punctuation ended it
        if current_sentence.strip():
            while sentence_chars and sentence_chars[0][0] in {"\r", "\n"}:
                sentence_chars.pop(0)
            while sentence_chars and sentence_chars[-1][0] in {"\r", "\n"}:
                sentence_chars.pop()
                        
            sentence_bbox = [
                min(b[0] for _, b in sentence_chars),  # x0
                min(b[1] for _, b in sentence_chars),  # y0
                max(b[2] for _, b in sentence_chars),  # x1
                max(b[3] for _, b in sentence_chars)   # y1
            ]
            
            results.append({
                "text": current_sentence.strip(),
                "bbox": sentence_bbox,
                "pageNumber": page_num
            })
    
    return {"results": results}
