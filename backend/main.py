import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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
import asyncio
import concurrent.futures
from functools import partial

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
    
    # Check PDF size limit (50MB = 52,428,800 bytes)
    if len(response.content) > 52_428_800:
        raise HTTPException(status_code=400, detail="PDF exceeds maximum size limit of 50MB")
    
    pdf_data = io.BytesIO(response.content)
    results = []
    sentence_enders = {".", "!", "?"}
    email_domains = {"com", "org", "edu"}

    # Open the PDF using PDFium
    pdf = pdfium.PdfDocument(pdf_data)
    
    # Check page count limit
    if len(pdf) > 2000:
        raise HTTPException(status_code=400, detail="PDF exceeds maximum page count of 2000")
    
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
    """Process PDF with OCR using parallel processing for performance"""
    results = []
    page_count = len(pdf)
    
    # Calculate optimal worker count based on available CPUs and page count
    cpu_count = min(multiprocessing.cpu_count(), 2)  # Respect container CPU limit
    worker_count = min(cpu_count, page_count)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First, render all pages to images in parallel
        render_tasks = []
        for page_num in range(page_count):
            render_tasks.append(render_page_to_image(pdf, page_num, temp_dir))
        
        # Wait for all renders to complete
        await asyncio.gather(*render_tasks)
        
        # Process OCR in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Create a list of tasks for OCR processing
            ocr_tasks = []
            for page_num in range(page_count):
                img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                page = pdf.get_page(page_num)
                page_width, page_height = page.get_size()
                
                # Process in smaller chunks for large documents
                if page_count > 20:
                    # Use lower scale for large documents
                    scale = 1.25
                else:
                    scale = 1.5
                
                ocr_task = loop.run_in_executor(
                    executor, 
                    process_page_ocr,
                    img_path, 
                    page_num, 
                    page_width, 
                    page_height,
                    scale
                )
                ocr_tasks.append(ocr_task)
            
            # Wait for all OCR tasks and collect results
            ocr_results = await asyncio.gather(*ocr_tasks)
            
            # Flatten results list
            for page_results in ocr_results:
                results.extend(page_results)
    
    return {"results": results}

async def render_page_to_image(pdf, page_num, temp_dir):
    """Render a PDF page to an image file"""
    # Lower scale for OCR to improve performance
    page = pdf.get_page(page_num)
    
    # Select appropriate scale based on page size
    page_width, page_height = page.get_size()
    page_area = page_width * page_height
    
    # Adjust scale based on page size (smaller scale for larger pages)
    if page_area > 1000000:  # Very large page
        scale = 1.2
    else:
        scale = 1.5
    
    # Run rendering in thread pool to prevent blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool,
            partial(render_page, pdf, page_num, temp_dir, scale)
        )
        
def render_page(pdf, page_num, temp_dir, scale):
    """Render function to run in thread pool"""
    page = pdf.get_page(page_num)
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()
    
    # Apply image preprocessing to improve OCR accuracy
    pil_image = preprocess_image(pil_image)
    
    # Save temporary image
    img_path = os.path.join(temp_dir, f"page_{page_num}.png")
    pil_image.save(img_path, quality=85)  # Slightly lower quality for speed
    
def preprocess_image(image):
    """Preprocess image to improve OCR quality and speed"""
    # Convert to grayscale for better OCR performance
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast to help text recognition
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    return image

def process_page_ocr(img_path, page_num, page_width, page_height, scale):
    """Process a single page with OCR - runs in a separate process"""
    page_results = []
    sentence_enders = {".", "!", "?"}
    
    # Load image
    pil_image = Image.open(img_path)
    
    # Configure tesseract for speed
    custom_config = r'--oem 1 --psm 6'
    
    # Perform OCR with pytesseract
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)
    
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
        if text.endswith(tuple(sentence_enders)) or ocr_data['conf'][i] < 50 or ocr_data.get('line_num', [0])[i] != ocr_data.get('line_num', [0])[i+1 if i+1 < len(ocr_data.get('line_num', [])) else i]:
            if current_sentence.strip():
                page_results.append({
                    "text": current_sentence.strip(),
                    "bbox": sentence_bbox,
                    "pageNumber": page_num
                })
                current_sentence = ""
                sentence_bbox = [100, 100, 0, 0]
    
    # Add any remaining text
    if current_sentence.strip():
        page_results.append({
            "text": current_sentence.strip(),
            "bbox": sentence_bbox,
            "pageNumber": page_num
        })
    
    return page_results