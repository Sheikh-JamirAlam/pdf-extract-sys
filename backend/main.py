from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import io
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
import tempfile
import os

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PDFRequest(BaseModel):
    pdf_url: str

# Add this function to detect if PDF needs OCR
def is_searchable_pdf(pdf):
    """Check if PDF has searchable text or needs OCR"""
    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        # If we find meaningful text on any page, consider it searchable
        if len(text.strip()) > 50:  # Arbitrary threshold
            return True
    return False

@app.post("/extract")
async def extract_pdf(pdf_req: PDFRequest):
    # Download the PDF
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_req.pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    
    pdf_data = io.BytesIO(response.content)
    results = []
    sentence_enders = {".", "!", "?"}
    email_domains = {"com", "org", "edu"}

    # Open the PDF using PDFium
    pdf = pdfium.PdfDocument(pdf_data)
    
    # Check if PDF needs OCR
    if not is_searchable_pdf(pdf):
        return await process_with_ocr(pdf, pdf_data)
    
    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)
        textpage = page.get_textpage()
        page_width, page_height = page.get_size()

        full_text = textpage.get_text_range()
        sentence_chars = []
        current_sentence = ""

        for index, char in enumerate(full_text):
            try:
                bbox = textpage.get_charbox(index, loose=False)
            except IndexError:
                continue

            if not bbox:
                continue

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
            if char == "." and ((index + 1 < len(full_text) and full_text[index + 1] == ".") or (index > 0 and full_text[index - 1] == ".")):
                continue
            if char == "." and ((index + 3 < len(full_text) and full_text[index + 1 : index + 3] == " .") or (index > 1 and full_text[index - 2 : index] == ". ")):
                continue
            # Check for emails
            if char == "." and (full_text[index + 1 : index + 4] in email_domains):
                continue
            # Check for URLs
            if char == "." and index > 3 and current_sentence[-4:].lower() == "www.":
                continue

            # If a sentence ends, store the text + bbox
            if char in sentence_enders or char == "\n" or char == "\ufffe":
                if current_sentence.strip():
                    
                    while sentence_chars and sentence_chars[0][0] in {"\r", "\n", " "}:
                        sentence_chars.pop(0)
                    while sentence_chars and sentence_chars[-1][0] in {"\r", "\n", " "}:
                        sentence_chars.pop()
                        
                    sentence_bbox = [
                        min(b[0] for _, b in sentence_chars),
                        min(b[1] for _, b in sentence_chars),
                        max(b[2] for _, b in sentence_chars),
                        max(b[3] for _, b in sentence_chars)
                    ]
                    
                    results.append({
                        "text": current_sentence.strip(),
                        "bbox": sentence_bbox,
                        "pageNumber": page_num
                    })
                    
                    current_sentence = ""
                    sentence_chars = []

        # Store remaining text
        if current_sentence.strip():
            while sentence_chars and sentence_chars[0][0] in {"\r", "\n", " "}:
                sentence_chars.pop(0)
            while sentence_chars and sentence_chars[-1][0] in {"\r", "\n", " "}:
                sentence_chars.pop()
                
            sentence_bbox = [
                min(b[0] for _, b in sentence_chars),
                min(b[1] for _, b in sentence_chars),
                max(b[2] for _, b in sentence_chars),
                max(b[3] for _, b in sentence_chars)
            ]
            results.append({"text": current_sentence.strip(), "bbox": sentence_bbox, "pageNumber": page_num})
    
    return {"results": results}

async def process_with_ocr(pdf, pdf_data):
    """Process PDF with OCR for non-searchable documents"""
    results = []
    sentence_enders = {".", "!", "?"}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # For each page, render to image and perform OCR
        for page_num in range(len(pdf)):
            page = pdf.get_page(page_num)
            bitmap = page.render(scale=2.0, rotation=0)
            pil_image = bitmap.to_pil()
            
            # Save temporary image
            img_path = os.path.join(temp_dir, f"page_{page_num}.png")
            pil_image.save(img_path)
            
            # Get page dimensions for bounding box normalization
            page_width, page_height = page.get_size()
            
            # Perform OCR with pytesseract
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            current_sentence = ""
            sentence_bbox = [100, 100, 0, 0]  # Initialize with extreme values
            
            # Process OCR results
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i]
                if not text.strip():
                    continue
                
                # Get coordinates and normalize
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Normalize bounding box
                x_norm = (x / pil_image.width) * 100
                y_norm = (y / pil_image.height) * 100
                right_norm = ((x + w) / pil_image.width) * 100
                bottom_norm = ((y + h) / pil_image.height) * 100
                
                # Update current sentence and its bounding box
                if current_sentence:
                    current_sentence += " "
                    # Expand the bounding box
                    sentence_bbox[0] = min(sentence_bbox[0], x_norm)
                    sentence_bbox[1] = min(sentence_bbox[1], y_norm)
                    sentence_bbox[2] = max(sentence_bbox[2], right_norm)
                    sentence_bbox[3] = max(sentence_bbox[3], bottom_norm)
                else:
                    # Initialize bounding box for new sentence
                    sentence_bbox = [x_norm, y_norm, right_norm, bottom_norm]
                
                current_sentence += text
                
                # Check if end of sentence or line
                if text.endswith(tuple(sentence_enders)) or ocr_data['conf'][i] < 50 or ocr_data['line_num'][i] != ocr_data.get('line_num', [0])[i+1 if i+1 < len(ocr_data['line_num']) else i]:
                    if current_sentence.strip():
                        results.append({
                            "text": current_sentence.strip(),
                            "bbox": sentence_bbox,
                            "pageNumber": page_num
                        })
                        current_sentence = ""
                        sentence_bbox = [100, 100, 0, 0]
            
            # Add any remaining text
            if current_sentence.strip():
                results.append({
                    "text": current_sentence.strip(),
                    "bbox": sentence_bbox,
                    "pageNumber": page_num
                })
    
    return {"results": results}