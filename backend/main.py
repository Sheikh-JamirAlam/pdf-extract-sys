from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import io
import pypdfium2 as pdfium

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

    # Open the PDF using PDFium
    pdf = pdfium.PdfDocument(pdf_data)
    
    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)
        textpage = page.get_textpage()
        page_width, page_height = page.get_size()

        current_sentence = ""
        sentence_chars = []  # Store characters and their bounding boxes

        # Extract the full text
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
                (bbox[0] / page_width) * 100,      # x0
                (1 - bbox[3] / page_height) * 100, # y0 (flipped)
                (bbox[2] / page_width) * 100,      # x1
                (1 - bbox[1] / page_height) * 100  # y1 (flipped)
            ]

            sentence_chars.append((char, bbox_normalized))
            current_sentence += char

            # If a sentence ender or line break is found, finalize sentence
            if char in sentence_enders or char == "\n":
                if current_sentence.strip():
                    # Compute bounding box from first to last character
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
