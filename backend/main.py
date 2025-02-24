from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
import pytesseract
import numpy as np

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

def is_searchable(page):
    """Check if a PDF page contains selectable text."""
    return bool(page.get_text().strip())

@app.post("/extract")
async def extract_pdf(pdf_req: PDFRequest):
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_req.pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    
    pdf_data = response.content
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    results = []
    sentence_enders = {".", "!", "?"}  # Sentence ending punctuation
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        
        if is_searchable(page):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    lines = block["lines"]
                    
                    for line in lines:
                        spans = line["spans"]
                        current_sentence = ""
                        sentence_bboxes = []
                        current_words = []

                        for span in spans:
                            text = span["text"].strip()
                            if not text:
                                continue
                            
                            bbox = span["bbox"]
                            sentence_bboxes.append(bbox)
                            words = text.split()
                            
                            # Compute per-word bounding boxes
                            span_width = bbox[2] - bbox[0]
                            char_width = span_width / len(text) if text else 0
                            
                            word_start = bbox[0]
                            for word in words:
                                word_width = len(word) * char_width
                                word_bbox = [
                                    word_start, bbox[1], word_start + word_width, bbox[3]
                                ]
                                current_words.append((word, word_bbox))
                                word_start += (len(word) + 1) * char_width  # Space handling
                                
                                # Add word to current sentence
                                current_sentence += word + " "

                                # If a sentence ender is found, finalize this sentence
                                if any(word.endswith(end) for end in sentence_enders):
                                    if current_sentence.strip():
                                        # Bounding box from first to last word
                                        first_word = current_words[0]
                                        last_word = current_words[-1]
                                        bbox_normalized = [
                                            (first_word[1][0] / page.rect.width) * 100,
                                            (first_word[1][1] / page.rect.height) * 100,
                                            (last_word[1][2] / page.rect.width) * 100,
                                            (last_word[1][3] / page.rect.height) * 100
                                        ]
                                        
                                        results.append({
                                            "text": current_sentence.strip(),
                                            "bbox": bbox_normalized,
                                            "pageNumber": page_num
                                        })
                                        
                                        # Reset sentence tracking
                                        current_sentence = ""
                                        current_words = []

                        # Handle remaining sentence if no punctuation ended it
                        if current_sentence.strip():
                            first_word = current_words[0]
                            last_word = current_words[-1]
                            bbox_normalized = [
                                (first_word[1][0] / page.rect.width) * 100,
                                (first_word[1][1] / page.rect.height) * 100,
                                (last_word[1][2] / page.rect.width) * 100,
                                (last_word[1][3] / page.rect.height) * 100
                            ]
                            
                            results.append({
                                "text": current_sentence.strip(),
                                "bbox": bbox_normalized,
                                "pageNumber": page_num
                            })
        else:
            # Perform OCR for scanned PDFs
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            ocr_text = pytesseract.image_to_string(img_data, config='--psm 6')
            
            results.append({
                "text": ocr_text.strip(),
                "bbox": [0, 0, 0, 0],  # No bounding box for OCR text
                "pageNumber": page_num
            })
    
    return {"results": results}
