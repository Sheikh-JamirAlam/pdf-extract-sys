from fastapi import FastAPI, HTTPException, BackgroundTasks
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
import time
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting implementation
class RateLimiter:
    def __init__(self, max_calls=1, period=1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = 0
        self.reset_time = time.time() + period
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            current_time = time.time()
            if current_time > self.reset_time:
                self.calls = 0
                self.reset_time = current_time + self.period
            
            if self.calls >= self.max_calls:
                wait_time = self.reset_time - current_time
                return False, wait_time
            
            self.calls += 1
            return True, 0
        
# Initialize rate limiter
rate_limiter = RateLimiter(max_calls=1, period=1.0)

class PDFRequest(BaseModel):
    pdf_url: str
    
class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    results: list = None
    message: str = None

# Dictionary to store processing jobs
processing_jobs = {}

# Add this function to detect if PDF needs OCR
def is_searchable_pdf(pdf, sample_pages=5):
    page_count = len(pdf)
    # Sample first, last, and middle pages for larger documents
    pages_to_check = min(sample_pages, page_count)
    
    if pages_to_check >= page_count:
        pages = range(page_count)
    else:
        # Sample evenly distributed pages
        step = max(1, page_count // pages_to_check)
        pages = [i * step for i in range(pages_to_check)]
        if (page_count - 1) not in pages:
            pages[-1] = page_count - 1  # Ensure we check the last page
    
    for page_num in pages:
        page = pdf.get_page(page_num)
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        # If we find meaningful text on any page, consider it searchable
        if len(text.strip()) > 50:  # Arbitrary threshold
            return True
    return False

def process_text_page(page, page_num):
    """Process a single page of text"""
    results = []
    textpage = page.get_textpage()
    page_width, page_height = page.get_size()
    full_text = textpage.get_text_range()
    
    sentence_enders = {".", "!", "?"}
    email_domains = {"com", "org", "edu", "net", "gov"}
    
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
        if char == "." and len(current_sentence) >= 2 and current_sentence[-2].isdigit() and index + 1 < len(full_text) and full_text[index + 1].isdigit():
            continue
            
        # Check for other special cases
        is_special_case = False
        
        # Check for mathematical sequences
        if char == "." and ((index + 1 < len(full_text) and full_text[index + 1] == ".") or (index > 0 and full_text[index - 1] == ".")):
            is_special_case = True
            
        # Check for emails
        if char == "." and index + 4 < len(full_text) and full_text[index + 1:index + 4].lower() in email_domains:
            is_special_case = True
            
        # Check for URLs
        if char == "." and index > 3 and current_sentence[-4:].lower() in ["www.", "http"]:
            is_special_case = True
            
        if is_special_case:
            continue

        # If a sentence ends, store the text + bbox
        if (char in sentence_enders or char == "\n" or char == "\ufffe") and not is_special_case:
            if current_sentence.strip():
                # Clean up start and end of sentence
                while sentence_chars and sentence_chars[0][0] in {"\r", "\n", " "}:
                    sentence_chars.pop(0)
                while sentence_chars and sentence_chars[-1][0] in {"\r", "\n", " "}:
                    sentence_chars.pop()
                    
                if sentence_chars:  # Make sure we still have characters
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
    if current_sentence.strip() and sentence_chars:
        while sentence_chars and sentence_chars[0][0] in {"\r", "\n", " "}:
            sentence_chars.pop(0)
        while sentence_chars and sentence_chars[-1][0] in {"\r", "\n", " "}:
            sentence_chars.pop()
            
        if sentence_chars:  # Make sure we still have characters
            sentence_bbox = [
                min(b[0] for _, b in sentence_chars),
                min(b[1] for _, b in sentence_chars),
                max(b[2] for _, b in sentence_chars),
                max(b[3] for _, b in sentence_chars)
            ]
            results.append({"text": current_sentence.strip(), "bbox": sentence_bbox, "pageNumber": page_num})
    
    return results

def process_ocr_page(pdf, page_num, temp_dir):
    """Process a single page with OCR"""
    results = []
    sentence_enders = {".", "!", "?"}
    
    page = pdf.get_page(page_num)
    # Reduce scale to 1.5 to improve performance while maintaining readable output
    bitmap = page.render(scale=1.5, rotation=0)
    pil_image = bitmap.to_pil()
    
    # Save temporary image
    img_path = os.path.join(temp_dir, f"page_{page_num}.png")
    pil_image.save(img_path, quality=85)  # Slightly reduced quality to save memory
    
    # Get page dimensions for bounding box normalization
    page_width, page_height = page.get_size()
    
    # Configure Tesseract for better performance
    custom_config = r'--oem 3 --psm 6'
    
    # Perform OCR with pytesseract
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)
    
    current_sentence = ""
    sentence_bbox = [100, 100, 0, 0]  # Initialize with extreme values
    current_line = -1
    
    # Process OCR results by grouping by line
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i]
        if not text.strip():
            continue
        
        # Get coordinates and normalize
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        line_num = ocr_data['line_num'][i]
        
        # Normalize bounding box
        x_norm = (x / pil_image.width) * 100
        y_norm = (y / pil_image.height) * 100
        right_norm = ((x + w) / pil_image.width) * 100
        bottom_norm = ((y + h) / pil_image.height) * 100
        
        # New line detection
        if line_num != current_line:
            # Save previous sentence if it exists
            if current_sentence.strip():
                results.append({
                    "text": current_sentence.strip(),
                    "bbox": sentence_bbox,
                    "pageNumber": page_num
                })
            # Reset for new line
            current_sentence = text
            sentence_bbox = [x_norm, y_norm, right_norm, bottom_norm]
            current_line = line_num
        else:
            # Continue current line
            current_sentence += " " + text
            # Expand the bounding box
            sentence_bbox[0] = min(sentence_bbox[0], x_norm)
            sentence_bbox[1] = min(sentence_bbox[1], y_norm)
            sentence_bbox[2] = max(sentence_bbox[2], right_norm)
            sentence_bbox[3] = max(sentence_bbox[3], bottom_norm)
        
        # Check for end of sentence within a line
        if text.endswith(tuple(sentence_enders)):
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
    
    # Clean up
    os.remove(img_path)
    return results

async def process_pdf(job_id, pdf_data):
    """Process PDF in background with progress tracking"""
    try:
        results = []
        start_time = time.time()
        
        # Open the PDF using PDFium
        pdf = pdfium.PdfDocument(pdf_data)
        page_count = len(pdf)
        
        logger.info(f"Job {job_id}: Processing PDF with {page_count} pages")
        processing_jobs[job_id]['status'] = f"Processing: 0/{page_count} pages"
        
        # Check if PDF needs OCR - sample a few pages instead of checking all
        needs_ocr = not is_searchable_pdf(pdf)
        logger.info(f"Job {job_id}: PDF needs OCR: {needs_ocr}")
        
        # Create a thread pool for processing
        max_workers = min(4, os.cpu_count() or 2)  # Limit thread count based on resource limits
        
        if needs_ocr:
            # Process with OCR
            with tempfile.TemporaryDirectory() as temp_dir:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    loop = asyncio.get_event_loop()
                    
                    # Process OCR in batches to control memory usage
                    batch_size = 10
                    for batch_start in range(0, page_count, batch_size):
                        batch_end = min(batch_start + batch_size, page_count)
                        tasks = []
                        
                        for page_num in range(batch_start, batch_end):
                            tasks.append(loop.run_in_executor(
                                executor, 
                                process_ocr_page, 
                                pdf, page_num, temp_dir
                            ))
                        
                        # Wait for all tasks in this batch to complete
                        batch_results = await asyncio.gather(*tasks)
                        for page_results in batch_results:
                            results.extend(page_results)
                        
                        # Update status
                        processing_jobs[job_id]['status'] = f"Processing: {batch_end}/{page_count} pages"
                        logger.info(f"Job {job_id}: Completed {batch_end}/{page_count} pages")
        else:
            # Process searchable PDF
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                loop = asyncio.get_event_loop()
                tasks = []
                
                # Process pages in batches
                batch_size = 50  # Larger batches for text extraction which is faster
                for batch_start in range(0, page_count, batch_size):
                    batch_end = min(batch_start + batch_size, page_count)
                    batch_tasks = []
                    
                    for page_num in range(batch_start, batch_end):
                        page = pdf.get_page(page_num)
                        batch_tasks.append(loop.run_in_executor(
                            executor,
                            process_text_page,
                            page, page_num
                        ))
                    
                    # Wait for all tasks in this batch to complete
                    batch_results = await asyncio.gather(*batch_tasks)
                    for page_results in batch_results:
                        results.extend(page_results)
                    
                    # Update status
                    processing_jobs[job_id]['status'] = f"Processing: {batch_end}/{page_count} pages"
                    logger.info(f"Job {job_id}: Completed {batch_end}/{page_count} pages")
        
        # Record completion time
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Job {job_id}: PDF processing completed in {processing_time:.2f} seconds")
        
        processing_jobs[job_id]['status'] = "completed"
        processing_jobs[job_id]['results'] = results
        
        # Clean up memory
        pdf = None
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error processing PDF: {str(e)}")
        processing_jobs[job_id]['status'] = "failed"
        processing_jobs[job_id]['message'] = str(e)

@app.post("/extract")
async def extract_pdf(pdf_req: PDFRequest, background_tasks: BackgroundTasks):
    # Apply rate limiting
    allowed, wait_time = await rate_limiter.acquire()
    if not allowed:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds"
        )
    
    # Download the PDF
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(pdf_req.pdf_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF")
        
        pdf_data = io.BytesIO(response.content)
        
        # Check file size (≤ 50MB)
        file_size = len(response.content) / (1024 * 1024)  # Size in MB
        if file_size > 50:
            raise HTTPException(status_code=400, detail=f"PDF size exceeds 50MB limit (got {file_size:.1f}MB)")
        
        # Generate a job ID
        job_id = f"job_{int(time.time() * 1000)}"
        processing_jobs[job_id] = {"status": "processing", "results": None}
        
        # Start processing in background
        background_tasks.add_task(process_pdf, job_id, pdf_data)
        
        return {"job_id": job_id, "status": "processing"}
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="PDF download timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    
    if job_data["status"] == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "results": job_data["results"]
        }
    elif job_data["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "message": job_data.get("message", "Unknown error")
        }
    else:
        return {
            "job_id": job_id,
            "status": job_data["status"]
        }

# Cleanup job for expired jobs (optional)
@app.on_event("startup")
@app.on_event("shutdown")
async def cleanup_expired_jobs():
    """Clean up old jobs to prevent memory leaks"""
    now = time.time()
    expired_time = 3600  # 1 hour
    
    expired_jobs = [
        job_id for job_id in list(processing_jobs.keys())
        if job_id.startswith("job_") and 
        (now - int(job_id.split("_")[1]) / 1000) > expired_time
    ]
    
    for job_id in expired_jobs:
        del processing_jobs[job_id]