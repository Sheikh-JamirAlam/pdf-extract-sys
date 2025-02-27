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
import numpy as np
import time
import cv2

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
    # Start timing
    start_time = time.time()
    
    try:
        # Download with timeout and size limit check
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(pdf_req.pdf_url)
            
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF")
        
        # Check size limit (50MB)
        if len(response.content) > 52_428_800:
            raise HTTPException(status_code=400, detail="PDF exceeds maximum size limit of 50MB")
        
        pdf_data = io.BytesIO(response.content)
        
        # Open PDF
        try:
            pdf = pdfium.PdfDocument(pdf_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {str(e)}")
        
        # Check page count
        page_count = len(pdf)
        if page_count > 2000:
            raise HTTPException(status_code=400, detail="PDF exceeds maximum page count of 2000")
        
        # Determine if PDF is searchable or needs OCR
        needs_ocr = not is_searchable_pdf(pdf)
        
        # Choose processing strategy
        if needs_ocr:
            # Use improved OCR processing
            results = await process_with_optimized_ocr(pdf, pdf_data)
        else:
            # For text PDFs, use optimized text extraction
            results = await extract_text(pdf, pdf_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processed {page_count} pages in {processing_time:.2f} seconds")
        
        # Add processing metadata
        results["processing_time"] = f"{processing_time:.2f} seconds"
        results["pages_processed"] = page_count
        results["ocr_used"] = needs_ocr
        
        return results
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        raise HTTPException(
            status_code=408, 
            detail=f"Processing timed out after {elapsed:.1f} seconds. For large documents, try processing fewer pages."
        )
    except Exception as e:
        elapsed = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail=f"Error after {elapsed:.1f} seconds: {str(e)}"
        )

async def extract_text(pdf, pdf_data):
    results = []
    sentence_enders = {".", "!", "?"}
    email_domains = {"com", "org", "edu"}
    
    tasks = []
    for page_num in range(len(pdf)):
        tasks.append(extract_page_text(pdf, page_num, sentence_enders, email_domains))
    
        # Process all pages
        page_results = await limit_concurrent_tasks(tasks, 3)
        
        # Flatten results
        for page_data in page_results:
            results.extend(page_data)
        
        return {"results": results}
    
async def limit_concurrent_tasks(tasks, limit=3):
    """Run tasks with a concurrency limit to prevent memory issues"""
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_limit(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(run_with_limit(task) for task in tasks))

async def extract_page_text(pdf, page_num, sentence_enders, email_domains):
    page_results = []
    
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
                
                page_results.append({
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
        page_results.append({"text": current_sentence.strip(), "bbox": sentence_bbox, "pageNumber": page_num})

    return page_results

async def process_with_optimized_ocr(pdf, pdf_data):
    """Process PDF with optimized OCR strategy for all pages"""
    results = []
    page_count = len(pdf)
    
    # Determine optimal batch size to keep memory usage low
    # batch_size = 8  # Process 10 pages at a time
    batch_size = max(5, min(20, int(75 / (page_count / 20))))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process all pages in batches
        for batch_start in range(0, page_count, batch_size):
            batch_end = min(batch_start + batch_size, page_count)
            
            # Force garbage collection between batches
            import gc
            gc.collect()
            
            # Clear previous batch files
            for f in os.listdir(temp_dir):
                if f.startswith("page_") and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except:
                        pass
            
            # Render current batch
            render_tasks = []
            for page_num in range(batch_start, batch_end):
                render_tasks.append(render_optimized_page(pdf, page_num, temp_dir))
            
            await asyncio.gather(*render_tasks)
            
            # Process OCR for current batch
            loop = asyncio.get_event_loop()
            worker_count = min(multiprocessing.cpu_count(), 2)  # Respect 2vCPU limit
            
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                ocr_tasks = []
                for page_num in range(batch_start, batch_end):
                    img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                    if not os.path.exists(img_path):
                        continue
                    
                    page = pdf.get_page(page_num)
                    page_width, page_height = page.get_size()
                    
                    ocr_task = loop.run_in_executor(
                        executor, 
                        process_page_with_optimized_ocr,
                        img_path, 
                        page_num, 
                        page_width, 
                        page_height
                    )
                    ocr_tasks.append((page_num, ocr_task))
                
                # Process with reasonable timeouts
                for page_num, future_task in ocr_tasks:
                    try:
                        page_results = await asyncio.wait_for(future_task, timeout=30)
                        results.extend(page_results)
                    except asyncio.TimeoutError:
                        results.append({
                            "text": f"[Timeout processing page {page_num}]",
                            "bbox": [0, 0, 100, 100],
                            "pageNumber": page_num
                        })
                    except Exception as e:
                        results.append({
                            "text": f"[Error processing page {page_num}]",
                            "bbox": [0, 0, 100, 100],
                            "pageNumber": page_num
                        })
    
    return {"results": results}

async def render_optimized_page(pdf, page_num, temp_dir):
    """Render a PDF page optimized for OCR"""
    page = pdf.get_page(page_num)
    
    # Higher scale for better OCR results
    scale = 2.0  # Higher DPI equivalent
    
    # Run rendering in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool,
            partial(render_page_with_preprocessing, pdf, page_num, temp_dir, scale)
        )
        
def render_page_with_preprocessing(pdf, page_num, temp_dir, scale):
    """Render page with image preprocessing for optimal OCR"""
    page = pdf.get_page(page_num)
    
    # Render at higher resolution
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()
    
    # Convert to OpenCV format for preprocessing
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Apply preprocessing optimized for OCR
    preprocessed = preprocess_image_for_ocr(open_cv_image)
    
    # Convert back to PIL and save
    pil_preprocessed = Image.fromarray(preprocessed)
    img_path = os.path.join(temp_dir, f"page_{page_num}.png")
    pil_preprocessed.save(img_path, format="PNG")
    
def preprocess_image_for_ocr(image):
    """Apply preprocessing techniques to improve OCR accuracy"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Check if the page is inverted (dark background with light text)
    is_inverted = np.mean(gray) < 127
    if is_inverted:
        gray = cv2.bitwise_not(gray)
    
    # Apply adaptive thresholding to handle varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Noise removal
    denoised = cv2.medianBlur(binary, 3)
    
    # Dilation to enhance text
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def process_page_with_optimized_ocr(img_path, page_num, page_width, page_height):
    """Process a single page with optimized OCR settings"""
    sentence_enders = {".", "!", "?"}
    
    try:
        # Load the preprocessed image
        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if cv_img is None:
            return [{
                "text": f"[Error loading image for page {page_num}]",
                "bbox": [0, 0, 100, 100],
                "pageNumber": page_num
            }]
        
        # Skip mostly blank pages
        if is_mostly_blank(cv_img, threshold=0.98):
            return []
        
        # Convert to PIL for Tesseract
        pil_image = Image.fromarray(cv_img)
        
        # Use optimized Tesseract configuration
        # Balance between speed and accuracy
        custom_config = r'--oem 1 --psm 6 -l eng --dpi 300'

        # Get both normal OCR text and structured data
        # text = pytesseract.image_to_string(pil_image, config=custom_config)
        ocr_data = pytesseract.image_to_data(
            pil_image, 
            output_type=pytesseract.Output.DICT, 
            config=custom_config
        )
        
        # Use more sophisticated sentence segmentation
        lines = []
        current_line = ""
        current_line_bbox = [100, 100, 0, 0]
        
        for i in range(len(ocr_data['text'])):
            # Skip low confidence or empty text
            if ocr_data['conf'][i] < 30 or not ocr_data['text'][i].strip():
                continue
                
            # Get word info
            word = ocr_data['text'][i]
            conf = ocr_data['conf'][i]
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            block_num = ocr_data['block_num'][i]
            line_num = ocr_data['line_num'][i]
            
            # Normalize bbox
            img_height, img_width = cv_img.shape
            x_norm = (x / img_width) * 100
            y_norm = (y / img_height) * 100
            right_norm = ((x + w) / img_width) * 100
            bottom_norm = ((y + h) / img_height) * 100
            
            # Update current sentence and its bounding box
            if current_line:
                current_line += " "
                # Expand the bounding box
                current_line_bbox[0] = min(current_line_bbox[0], x_norm)
                current_line_bbox[1] = min(current_line_bbox[1], y_norm)
                current_line_bbox[2] = max(current_line_bbox[2], right_norm)
                current_line_bbox[3] = max(current_line_bbox[3], bottom_norm)
            else:
                # Initialize bounding box for new sentence
                current_line_bbox = [x_norm, y_norm, right_norm, bottom_norm]
            
            current_line += word
            
            # Check if end of sentence or line
            if word.endswith(tuple(sentence_enders)) or ocr_data['conf'][i] < 50 or ocr_data['line_num'][i] != ocr_data.get('line_num', [0])[i+1 if i+1 < len(ocr_data['line_num']) else i]:
                if current_line.strip():
                    lines.append({
                        "text": current_line.strip(),
                        "bbox": current_line_bbox,
                        "pageNumber": page_num
                    })
                    current_line = ""
                    current_line_bbox = [100, 100, 0, 0]
            
        # Add any remaining text
        if current_line.strip():
            lines.append({
                "text": current_line.strip(),
                "bbox": current_line_bbox,
                "pageNumber": page_num
            })
                
        return lines
        
    except Exception as e:
        print(f"Error in optimized OCR: {e}")
        return [{
            "text": f"[Error processing page {page_num}]",
            "bbox": [0, 0, 100, 100],
            "pageNumber": page_num
        }]

def is_mostly_blank(image, threshold=0.95):
    """Check if an image is mostly blank/white space"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Count pixels that are close to white
    white_pixels = np.sum(image > 220)
    total_pixels = image.shape[0] * image.shape[1]
    
    return white_pixels / total_pixels > threshold
