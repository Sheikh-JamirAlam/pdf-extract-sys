import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import io
import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
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
    """Process PDF with optimized OCR strategy"""
    results = []
    page_count = len(pdf)
    
    # Determine optimal processing strategy based on PDF size
    cpu_count = min(multiprocessing.cpu_count(), 2)  # Respect container limit
    worker_count = min(cpu_count, page_count, 1 if page_count > 50 else 2)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # For very large documents, sample strategically
        if page_count > 150:
            return await process_with_smart_sampling(pdf, pdf_data, page_count, temp_dir)

        # Otherwise process all pages with optimized OCR
        render_tasks = []
        for page_num in range(page_count):
            render_tasks.append(render_optimized_page(pdf, page_num, temp_dir))
        
        # Wait for all renders to complete
        await asyncio.gather(*render_tasks)
        
        # Process OCR in parallel with optimized settings
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            ocr_tasks = []
            for page_num in range(page_count):
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
                ocr_tasks.append(ocr_task)
            
            # Process results
            for future in asyncio.as_completed(ocr_tasks, timeout=60):
                try:
                    page_results = await future
                    results.extend(page_results)
                except Exception as e:
                    print(f"Error in OCR processing: {e}")
    
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
        text = pytesseract.image_to_string(pil_image, config=custom_config)
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
        
        # # Use more sophisticated sentence segmentation
        # lines = []
        # current_line = []
        # current_line_bbox = [100, 100, 0, 0]
        
        # for i in range(len(ocr_data['text'])):
        #     # Skip low confidence or empty text
        #     if ocr_data['conf'][i] < 30 or not ocr_data['text'][i].strip():
        #         continue
                
        #     # Get word info
        #     word = ocr_data['text'][i]
        #     conf = ocr_data['conf'][i]
        #     x = ocr_data['left'][i]
        #     y = ocr_data['top'][i]
        #     w = ocr_data['width'][i]
        #     h = ocr_data['height'][i]
        #     block_num = ocr_data['block_num'][i]
        #     line_num = ocr_data['line_num'][i]
            
        #     # Normalize bbox
        #     img_height, img_width = cv_img.shape
        #     x_norm = (x / img_width) * 100
        #     y_norm = (y / img_height) * 100
        #     right_norm = ((x + w) / img_width) * 100
        #     bottom_norm = ((y + h) / img_height) * 100
            
        #     # Check if we're on a new line
        #     if current_line and (block_num != ocr_data['block_num'][i-1] or 
        #                         line_num != ocr_data['line_num'][i-1]):
        #         # Save the completed line
        #         line_text = " ".join(current_line).strip()
        #         if line_text:
        #             lines.append({
        #                 "text": line_text,
        #                 "bbox": current_line_bbox,
        #                 "pageNumber": page_num
        #             })
        #         # Reset for new line
        #         current_line = []
        #         current_line_bbox = [100, 100, 0, 0]
            
        #     # Add word to current line
        #     current_line.append(word)
            
        #     # Update the line's bounding box
        #     current_line_bbox[0] = min(current_line_bbox[0], x_norm)
        #     current_line_bbox[1] = min(current_line_bbox[1], y_norm)
        #     current_line_bbox[2] = max(current_line_bbox[2], right_norm)
        #     current_line_bbox[3] = max(current_line_bbox[3], bottom_norm)
            
        #     print(f"current_line {current_line}")
        
        # # Don't forget the last line
        # if current_line:
        #     line_text = " ".join(current_line).strip()
        #     if line_text:
        #         lines.append({
        #             "text": line_text,
        #             "bbox": current_line_bbox,
        #             "pageNumber": page_num
        #         })
        
        # #Now group lines into paragraphs - improve semantic organization
        # paragraphs = []
        # current_paragraph = []
        # current_para_bbox = [100, 100, 0, 0]
        
        # for line in lines:
        #     # Simple paragraph detection based on line spacing and indentation
        #     if current_paragraph:
        #         # Get approximate line spacing
        #         last_line_bottom = current_para_bbox[3]
        #         this_line_top = line["bbox"][1]
                
        #         # If there's a significant gap, start a new paragraph
        #         if (this_line_top - last_line_bottom) > 1.5:  # Threshold for paragraph break
        #             # Save current paragraph
        #             para_text = " ".join([p["text"] for p in current_paragraph])
        #             paragraphs.append({
        #                 "text": para_text,
        #                 "bbox": current_para_bbox,
        #                 "pageNumber": page_num
        #             })
        #             # Reset
        #             current_paragraph = []
        #             current_para_bbox = [100, 100, 0, 0]
            
        #     # Add line to current paragraph
        #     current_paragraph.append(line)
            
        #     # Update paragraph bbox
        #     line_bbox = line["bbox"]
        #     current_para_bbox[0] = min(current_para_bbox[0], line_bbox[0])
        #     current_para_bbox[1] = min(current_para_bbox[1], line_bbox[1])
        #     current_para_bbox[2] = max(current_para_bbox[2], line_bbox[2])
        #     current_para_bbox[3] = max(current_para_bbox[3], line_bbox[3])
        
        # # Add final paragraph
        # if current_paragraph:
        #     para_text = " ".join([p["text"] for p in current_paragraph])
        #     paragraphs.append({
        #         "text": para_text,
        #         "bbox": current_para_bbox,
        #         "pageNumber": page_num
        #     })
        
        # return paragraphs
        
    except Exception as e:
        print(f"Error in optimized OCR: {e}")
        return [{
            "text": f"[Error processing page {page_num}]",
            "bbox": [0, 0, 100, 100],
            "pageNumber": page_num
        }]

async def process_with_smart_sampling(pdf, pdf_data, page_count, temp_dir):
    """Smart sampling strategy for large documents"""
    results = []
    
    # Determine sample size based on document size
    # The sampling rate adapts based on page count to stay within time constraints
    if page_count > 500:
        sample_rate = max(5, min(20, int(page_count / 100)))  # 5-20% sampling
    else:
        sample_rate = max(10, min(30, int(page_count / 50)))  # 10-30% sampling
    
    # Smart sampling strategy:
    sample_pages = []
    
    # Always include the first 5 pages (typically TOC, abstract, etc.)
    sample_pages.extend(list(range(min(5, page_count))))
    
    # Always include the last 3 pages (references, conclusions)
    if page_count > 8:
        sample_pages.extend(list(range(max(0, page_count-3), page_count)))
    
    # Sample evenly throughout the document
    step = max(1, int(page_count / sample_rate))
    sample_pages.extend(list(range(5, page_count-3, step)))
    
    # Add random samples to catch important content that might be missed
    import random
    random.seed(42)  # For reproducibility
    remaining_pages = set(range(page_count)) - set(sample_pages)
    if remaining_pages:
        random_sample_count = min(len(remaining_pages), int(page_count * 0.05))  # 5% random sampling
        random_pages = random.sample(list(remaining_pages), random_sample_count)
        sample_pages.extend(random_pages)
    
    # Remove duplicates and sort
    sample_pages = sorted(list(set(sample_pages)))
    
    # Render and process sample pages with high quality
    render_tasks = []
    for page_num in sample_pages:
        render_tasks.append(render_optimized_page(pdf, page_num, temp_dir))
    
    await asyncio.gather(*render_tasks)
    
    # Process OCR for sample pages with high quality settings
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=2) as executor:
        ocr_tasks = []
        for page_num in sample_pages:
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
            ocr_tasks.append(ocr_task)
        
        # Process results
        for future in asyncio.as_completed(ocr_tasks, timeout=90):
            try:
                page_results = await future
                results.extend(page_results)
            except Exception as e:
                print(f"Error in smart sampling OCR: {e}")
    
    # Add metadata to indicate sampling was used
    return {
        "results": results, 
        "sampling_used": True,
        "pages_processed": len(sample_pages),
        "total_pages": page_count,
        "sample_rate": f"{len(sample_pages)}/{page_count}"
    }

def is_mostly_blank(image, threshold=0.95):
    """Check if an image is mostly blank/white space"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Count pixels that are close to white
    white_pixels = np.sum(image > 220)
    total_pixels = image.shape[0] * image.shape[1]
    
    return white_pixels / total_pixels > threshold

# @app.post("/extract")
# async def extract_pdf(pdf_req: PDFRequest):
#     # Start timing
#     start_time = time.time()
    
#     # Add support for progress tracking
#     processing_status = {"status": "downloading", "progress": 0}
#     try:
#         # Download with timeout and size limit check
#         async with httpx.AsyncClient(timeout=15) as client:
#             response = await client.get(pdf_req.pdf_url)
            
#         if response.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download PDF")
        
#         # Check size limit (50MB)
#         if len(response.content) > 52_428_800:
#             raise HTTPException(status_code=400, detail="PDF exceeds maximum size limit of 50MB")
        
#         processing_status["status"] = "analyzing"
#         processing_status["progress"] = 10
        
#         pdf_data = io.BytesIO(response.content)
        
#         # Open PDF
#         try:
#             pdf = pdfium.PdfDocument(pdf_data)
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid PDF: {str(e)}")
        
#         # Check page count
#         page_count = len(pdf)
#         if page_count > 2000:
#             raise HTTPException(status_code=400, detail="PDF exceeds maximum page count of 2000")
        
#         processing_status["status"] = f"processing {page_count} pages"
#         processing_status["progress"] = 20
        
#         # CRITICAL: Estimate processing time and choose strategy
#         if page_count > 170:
#             # ULTRA-AGGRESSIVE: For very large PDFs, we must sample
#             if not is_searchable_pdf(pdf):
#                 results = await process_with_sampling(pdf, pdf_data, page_count)
#             else:
#                 # For text PDFs, we can still process all pages
#                 results = await extract_text_optimized(pdf, pdf_data)
#         elif page_count > 50:
#             # AGGRESSIVE: For medium-large PDFs
#             if not is_searchable_pdf(pdf):
#                 results = await process_with_ocr(pdf, pdf_data, fast_mode=True)
#             else:
#                 results = await extract_text_optimized(pdf, pdf_data)
#         else:
#             # NORMAL: For smaller PDFs
#             if not is_searchable_pdf(pdf):
#                 results = await process_with_ocr(pdf, pdf_data, fast_mode=False)
#             else:
#                 results = await extract_text(pdf)
        
#         # Calculate processing time
#         processing_time = time.time() - start_time
#         print(f"Processed {page_count} pages in {processing_time:.2f} seconds")
        
#         # Add processing metadata
#         results["processing_time"] = f"{processing_time:.2f} seconds"
#         results["pages_processed"] = page_count
        
#         return results
        
#     except asyncio.TimeoutError:
#         elapsed = time.time() - start_time
#         raise HTTPException(
#             status_code=408, 
#             detail=f"Processing timed out after {elapsed:.1f} seconds. For large documents, try processing fewer pages."
#         )
#     except Exception as e:
#         elapsed = time.time() - start_time
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error after {elapsed:.1f} seconds: {str(e)}"
#         )

# def process_page_ocr(img_path, page_num, page_width, page_height, scale):
#     """Process a single page with OCR - runs in a separate process"""
#     page_results = []
#     sentence_enders = {".", "!", "?"}
    
#     # Load image
#     pil_image = Image.open(img_path)
    
#     # Configure tesseract for speed
#     custom_config = r'--oem 1 --psm 6'
    
#     # Perform OCR with pytesseract
#     ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)
    
#     current_sentence = ""
#     sentence_bbox = [100, 100, 0, 0]  # Initialize with extreme values
    
#     # Process OCR results
#     for i in range(len(ocr_data['text'])):
#         text = ocr_data['text'][i]
#         if not text.strip():
#             continue
        
#         # Get coordinates and normalize
#         x = ocr_data['left'][i]
#         y = ocr_data['top'][i]
#         w = ocr_data['width'][i]
#         h = ocr_data['height'][i]
        
#         # Normalize bounding box
#         x_norm = (x / pil_image.width) * 100
#         y_norm = (y / pil_image.height) * 100
#         right_norm = ((x + w) / pil_image.width) * 100
#         bottom_norm = ((y + h) / pil_image.height) * 100
        
#         # Update current sentence and its bounding box
#         if current_sentence:
#             current_sentence += " "
#             # Expand the bounding box
#             sentence_bbox[0] = min(sentence_bbox[0], x_norm)
#             sentence_bbox[1] = min(sentence_bbox[1], y_norm)
#             sentence_bbox[2] = max(sentence_bbox[2], right_norm)
#             sentence_bbox[3] = max(sentence_bbox[3], bottom_norm)
#         else:
#             # Initialize bounding box for new sentence
#             sentence_bbox = [x_norm, y_norm, right_norm, bottom_norm]
        
#         current_sentence += text
        
#         # Check if end of sentence or line
#         if text.endswith(tuple(sentence_enders)) or ocr_data['conf'][i] < 50 or ocr_data.get('line_num', [0])[i] != ocr_data.get('line_num', [0])[i+1 if i+1 < len(ocr_data.get('line_num', [])) else i]:
#             if current_sentence.strip():
#                 page_results.append({
#                     "text": current_sentence.strip(),
#                     "bbox": sentence_bbox,
#                     "pageNumber": page_num
#                 })
#                 current_sentence = ""
#                 sentence_bbox = [100, 100, 0, 0]
    
#     # Add any remaining text
#     if current_sentence.strip():
#         page_results.append({
#             "text": current_sentence.strip(),
#             "bbox": sentence_bbox,
#             "pageNumber": page_num
#         })
    
#     return page_results

# # 1. SAMPLING APPROACH: Process only a subset of pages
# async def process_with_ocr(pdf, pdf_data, fast_mode=False):
#     """Process PDF with aggressive optimizations for large documents"""
#     results = []
#     page_count = len(pdf)
    
#     # CRITICAL SPEED OPTIMIZATION FOR LARGE DOCUMENTS
#     if page_count > 100:
#         # Use intelligent sampling for very large documents
#         return await process_with_sampling(pdf, pdf_data, page_count)
    
#     # For smaller documents, use the regular parallel approach but with optimizations
#     cpu_count = min(multiprocessing.cpu_count(), 2)  # Respect container limit
#     worker_count = min(cpu_count, page_count)
    
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Render pages in parallel with lower quality
#         render_tasks = []
#         for page_num in range(page_count):
#             render_tasks.append(render_page_to_image(pdf, page_num, temp_dir, fast_mode=True))
        
#         # Wait for renders to complete
#         await asyncio.gather(*render_tasks)
        
#         # Process OCR in parallel
#         loop = asyncio.get_event_loop()
#         with ProcessPoolExecutor(max_workers=worker_count) as executor:
#             ocr_tasks = []
#             for page_num in range(page_count):
#                 img_path = os.path.join(temp_dir, f"page_{page_num}.png")
#                 page = pdf.get_page(page_num)
#                 page_width, page_height = page.get_size()
                
#                 # Use fast mode settings
#                 ocr_task = loop.run_in_executor(
#                     executor, 
#                     process_page_ocr_fast,  # Using faster OCR function
#                     img_path, 
#                     page_num, 
#                     page_width, 
#                     page_height
#                 )
#                 ocr_tasks.append(ocr_task)
            
#             # Gather results with timeout protection
#             for future in asyncio.as_completed(ocr_tasks, timeout=60):
#                 try:
#                     page_results = await future
#                     results.extend(page_results)
#                 except Exception as e:
#                     print(f"Error processing page: {e}")
    
#     return {"results": results}

# # 2. SAMPLING APPROACH: Only process a strategic subset of pages
# async def process_with_sampling(pdf, pdf_data, page_count):
#     """Use intelligent sampling for very large documents"""
#     results = []
    
#     # Strategic sampling - process:
#     # - First 10 pages (usually important content)
#     # - Last 5 pages (often contains conclusions/references)
#     # - Every 10th page in between
#     sample_pages = list(range(min(10, page_count)))  # First 10 pages
    
#     if page_count > 15:
#         sample_pages.extend(list(range(max(0, page_count-5), page_count)))  # Last 5 pages
        
#     # Add every 10th page in between
#     if page_count > 20:
#         sample_pages.extend(list(range(10, page_count-5, 10)))
    
#     # Remove any duplicates and sort
#     sample_pages = sorted(list(set(sample_pages)))
    
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Render only sample pages
#         render_tasks = []
#         for page_num in sample_pages:
#             render_tasks.append(render_page_to_image(pdf, page_num, temp_dir, fast_mode=True))
        
#         await asyncio.gather(*render_tasks)
        
#         # Process OCR for sample pages
#         loop = asyncio.get_event_loop()
#         with ProcessPoolExecutor(max_workers=2) as executor:
#             ocr_tasks = []
#             for page_num in sample_pages:
#                 img_path = os.path.join(temp_dir, f"page_{page_num}.png")
#                 if not os.path.exists(img_path):
#                     continue
                    
#                 page = pdf.get_page(page_num)
#                 page_width, page_height = page.get_size()
                
#                 ocr_task = loop.run_in_executor(
#                     executor, 
#                     process_page_ocr_fast,
#                     img_path, 
#                     page_num, 
#                     page_width, 
#                     page_height
#                 )
#                 ocr_tasks.append(ocr_task)
            
#             # Process results
#             for future in asyncio.as_completed(ocr_tasks, timeout=60):
#                 try:
#                     page_results = await future
#                     results.extend(page_results)
#                 except Exception as e:
#                     print(f"Error in sampled OCR: {e}")
    
#     # Add metadata to indicate sampling was used
#     return {
#         "results": results, 
#         "sampling_used": True,
#         "pages_processed": len(sample_pages),
#         "total_pages": page_count,
#         "sample_rate": f"{len(sample_pages)}/{page_count}"
#     }

# # 3. ULTRA-FAST RENDERING with minimal quality
# async def render_page_to_image(pdf, page_num, temp_dir, fast_mode=False):
#     """Render a PDF page to image with speed optimization"""
#     page = pdf.get_page(page_num)
    
#     # SPEED OPTIMIZATION: Scale based on fast_mode
#     scale = 0.8 if fast_mode else 1.2
    
#     # Run rendering in thread pool
#     loop = asyncio.get_event_loop()
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         await loop.run_in_executor(
#             pool,
#             partial(render_page_ultrafast, pdf, page_num, temp_dir, scale)
#         )

# def render_page_ultrafast(pdf, page_num, temp_dir, scale):
#     """Ultra-fast rendering with minimal preprocessing"""
#     page = pdf.get_page(page_num)
    
#     # SPEED OPTIMIZATION: Render at lower quality
#     bitmap = page.render(scale=scale, rotation=0)
#     pil_image = bitmap.to_pil()
    
#     # SPEED OPTIMIZATION: Aggressive image simplification
#     if pil_image.mode != 'L':
#         pil_image = pil_image.convert('L')
    
#     # SPEED OPTIMIZATION: Minimal enhancement, just enough for OCR
#     pil_image = pil_image.filter(ImageFilter.SHARPEN)
    
#     # SPEED OPTIMIZATION: Lower quality save
#     img_path = os.path.join(temp_dir, f"page_{page_num}.png")
#     pil_image.save(img_path, format="PNG", optimize=True, quality=65)

# # 4. ULTRA-FAST OCR PROCESSING
# def process_page_ocr_fast(img_path, page_num, page_width, page_height):
#     """Ultra-fast OCR processing with minimal accuracy requirements"""
#     page_results = []
    
#     # Load image
#     try:
#         pil_image = Image.open(img_path)
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         return []
    
#     # SPEED OPTIMIZATION: Configure tesseract for maximum speed
#     custom_config = r'--oem 1 --psm 6 -l eng --dpi 150'
    
#     try:
#         # SPEED OPTIMIZATION: Try to use page segmentation to only process text regions
#         # This can dramatically reduce processing time for pages with sparse text
#         # Use OpenCV to detect text regions
#         cv_img = np.array(pil_image)
        
#         # SPEED OPTIMIZATION: Resize large images
#         h, w = cv_img.shape[:2]
#         if max(h, w) > 2000:
#             scale_factor = 2000 / max(h, w)
#             new_w = int(w * scale_factor)
#             new_h = int(h * scale_factor)
#             cv_img = cv2.resize(cv_img, (new_w, new_h))
#             pil_image = Image.fromarray(cv_img)
        
#         # SPEED OPTIMIZATION: Skip mostly blank pages
#         if is_mostly_blank(cv_img):
#             return []
            
#         # Perform OCR with aggressive timeout
#         start_time = time.time()
#         ocr_data = pytesseract.image_to_data(
#             pil_image, 
#             output_type=pytesseract.Output.DICT, 
#             config=custom_config
#         )
        
#         # SPEED OPTIMIZATION: If OCR takes too long, return partial results
#         if time.time() - start_time > 3:  # 3 second timeout per page
#             print(f"OCR timeout on page {page_num}")
#             return [{
#                 "text": f"[OCR timeout on page {page_num}]",
#                 "bbox": [0, 0, 100, 100],
#                 "pageNumber": page_num
#             }]
        
#         # Process OCR results
#         current_sentence = ""
#         sentence_bbox = [100, 100, 0, 0]
#         sentence_enders = {".", "!", "?"}
        
#         # Simplified text extraction that focuses on speed over accuracy
#         for i in range(len(ocr_data['text'])):
#             if ocr_data['conf'][i] < 40:  # Skip low confidence results
#                 continue
                
#             text = ocr_data['text'][i].strip()
#             if not text:
#                 continue
            
#             # Get coordinates and normalize
#             x = ocr_data['left'][i]
#             y = ocr_data['top'][i]
#             w = ocr_data['width'][i]
#             h = ocr_data['height'][i]
            
#             # Normalize bounding box
#             img_width, img_height = pil_image.size
#             x_norm = (x / img_width) * 100
#             y_norm = (y / img_height) * 100
#             right_norm = ((x + w) / img_width) * 100
#             bottom_norm = ((y + h) / img_height) * 100
            
#             # Update current sentence
#             if current_sentence:
#                 current_sentence += " "
#                 sentence_bbox[0] = min(sentence_bbox[0], x_norm)
#                 sentence_bbox[1] = min(sentence_bbox[1], y_norm)
#                 sentence_bbox[2] = max(sentence_bbox[2], right_norm)
#                 sentence_bbox[3] = max(sentence_bbox[3], bottom_norm)
#             else:
#                 sentence_bbox = [x_norm, y_norm, right_norm, bottom_norm]
            
#             current_sentence += text
            
#             # Check for sentence end
#             if (text.endswith(tuple(sentence_enders)) or 
#                 i == len(ocr_data['text']) - 1 or
#                 ocr_data.get('block_num', [0])[i] != ocr_data.get('block_num', [0])[i+1 if i+1 < len(ocr_data.get('block_num', [])) else i]):
                
#                 if current_sentence.strip():
#                     page_results.append({
#                         "text": current_sentence.strip(),
#                         "bbox": sentence_bbox,
#                         "pageNumber": page_num
#                     })
#                 current_sentence = ""
#                 sentence_bbox = [100, 100, 0, 0]
        
#         # Add any remaining text
#         if current_sentence.strip():
#             page_results.append({
#                 "text": current_sentence.strip(),
#                 "bbox": sentence_bbox,
#                 "pageNumber": page_num
#             })
            
#         return page_results
        
#     except Exception as e:
#         print(f"Error in OCR: {e}")
#         return [{
#             "text": f"[Error processing page {page_num}]",
#             "bbox": [0, 0, 100, 100],
#             "pageNumber": page_num
#         }]
