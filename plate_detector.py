"""
Vehicle Number Plate Detection Module
Contains plate detection, localization, and preprocessing functions
Enhanced for MOBILE IMAGE DETECTION and INSTANT DETECTION
"""
import cv2
import numpy as np
import easyocr
from imutils import resize

# Initialize OCR reader (lazy loading)
_reader = None

# Stability tracking
_plate_history = []
_history_size = 5

def get_reader():
    """Lazy load OCR reader"""
    global _reader
    if _reader is None:
        print("Loading EasyOCR reader...", flush=True)
        try:
            _reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        except:
            _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("EasyOCR ready!", flush=True)
    return _reader


def get_stable_plate(text, conf):
    """
    Apply stability check - require same plate detected multiple times
    """
    global _plate_history, _history_size
    
    if not text or text == "No plate detected":
        return text, conf
    
    # Add to history
    _plate_history.append((text, conf))
    
    # Keep only recent history
    if len(_plate_history) > _history_size:
        _plate_history.pop(0)
    
    # Need at least 2 detections to be stable
    if len(_plate_history) < 2:
        return None, 0
    
    # Check if we have consistent results
    texts = [t for t, c in _plate_history if t]
    if not texts:
        return None, 0
    
    # Find most common plate
    from collections import Counter
    most_common = Counter(texts).most_common(1)[0]
    
    # Need at least 2 occurrences of the same plate
    if most_common[1] >= 2:
        # Get average confidence for this plate
        confs = [c for t, c in _plate_history if t == most_common[0]]
        avg_conf = sum(confs) / len(confs)
        
        # Clear history after stable detection
        _plate_history.clear()
        
        print(f"STABLE DETECTION: {most_common[0]} (avg conf: {avg_conf:.2f})", flush=True)
        return most_common[0], avg_conf
    
    return None, 0


def preprocess_image(image):
    """
    Preprocess image for better plate detection
    - Convert to grayscale
    - Apply CLAHE for contrast enhancement
    - Apply bilateral filter for noise reduction
    """
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return gray, enhanced, thresh


def detect_plate_contours(enhanced):
    """
    Detect potential license plate regions using contour detection
    """
    # Morphological operations to find rectangular regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    
    # Find edges
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    plate_candidates = []
    
    for contour in contours[:50]:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (plates are typically 2:1 to 5:1)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        if 1.5 < aspect_ratio < 6 and area > 1000:
            # This could be a license plate
            plate_candidates.append((x, y, w, h))
    
    return plate_candidates


def detect_plate_color(image):
    """
    Detect plate using color segmentation
    Looks for typical plate colors (white, yellow, blue)
    """
    if len(image.shape) != 3:
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for plates
    color_ranges = [
        # White plates
        ((0, 0, 150), (180, 50, 255)),
        # Yellow plates
        ((20, 50, 50), (30, 255, 255)),
        # Blue plates
        ((100, 50, 50), (130, 255, 255)),
    ]
    
    for (lower, upper) in color_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 1.5 < aspect_ratio < 6 and w > 50 and h > 20:
                return (x, y, w, h)
    
    return None


def apply_haar_cascade(gray, cascade):
    """
    Apply Haar cascade for plate detection
    """
    plates = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 10),
        maxSize=(300, 100)
    )
    
    if len(plates) > 0:
        # Return the largest detection
        largest = max(plates, key=lambda x: x[2] * x[3])
        return tuple(largest)
    
    return None


def extract_plate_region(image, region):
    """
    Extract and enhance the plate region from the image
    """
    x, y, w, h = region
    
    # Add some padding
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2*pad)
    h = min(image.shape[0] - y, h + 2*pad)
    
    # Extract region
    plate_region = image[y:y+h, x:x+w]
    
    if plate_region.size == 0:
        return None
    
    # Resize for better OCR
    plate_region = cv2.resize(plate_region, (0, 0), fx=2, fy=2)
    
    return plate_region


def detect_and_read_plate(image):
    """
    Main function: Detect license plate and read text
    Returns: (text, confidence, debug_image)
    """
    if image is None:
        return "No image provided", 0, None
    
    debug_img = image.copy()
    reader = get_reader()
    
    # Step 1: Preprocess
    gray, enhanced, thresh = preprocess_image(image)
    
    # Step 2: Try multiple detection methods
    plate_region = None
    detection_method = None
    
    # Method 1: Direct OCR on resized image (often most reliable)
    # Resize for better OCR
    resized = cv2.resize(gray, (640, 480))
    
    print("Trying enhanced full image OCR...", flush=True)
    try:
        results = reader.readtext(
            resized,
            contrast_ths=0.3,
            adjust_contrast=0.5,
            text_threshold=0.6,
            width_ths=0.5
        )
        
        print(f"OCR found {len(results)} results", flush=True)
        
        best_text = ""
        best_conf = 0
        
        for result in results:
            if len(result) >= 3:
                bbox, text, conf = result[0], result[1], result[2]
            elif len(result) >= 2:
                text = result[0] if isinstance(result[0], str) else result[1]
                conf = result[1] if isinstance(result[1], (int, float)) else 0.5
            else:
                continue
            
            text = str(text).strip()
            if not text:
                continue
            
            # Clean the text - keep only valid plate characters
            cleaned = ''.join(c for c in text.upper() if c.isalnum() or c in '- ')
            cleaned = cleaned.replace(' ', '')
            
            # Skip very short or very long text
            if len(cleaned) < 4 or len(cleaned) > 15:
                continue
            
            # Normalize confidence
            conf = float(conf) if conf is not None else 0.5
            if conf > 1:
                conf = conf / 100.0
            
            # Check if it looks like a real license plate
            # Common patterns: 
            # - State code + number + letters: MH12AB1234
            # - Just numbers: 1234
            # - Letters + numbers: ABC123
            
            has_letters = any(c.isalpha() for c in cleaned)
            has_numbers = any(c.isdigit() for c in cleaned)
            
            # Valid plate: be more lenient - any combination of letters and numbers
            is_valid_plate = False
            
            if len(cleaned) >= 4:
                has_letters = any(c.isalpha() for c in cleaned)
                has_numbers = any(c.isdigit() for c in cleaned)
                
                # More lenient: accept if has both OR has numbers with 4-6 digits
                if has_letters and has_numbers:
                    is_valid_plate = True
                elif has_numbers and len(cleaned) >= 4 and len(cleaned) <= 8:
                    is_valid_plate = True
                elif has_letters and len(cleaned) >= 4 and len(cleaned) <= 6:
                    is_valid_plate = True
            
            print(f"  Candidate: '{cleaned}' conf: {conf:.2f}, valid: {is_valid_plate}", flush=True)
            
            if is_valid_plate and conf > best_conf:
                best_text = cleaned
                best_conf = conf
        
        if best_text and best_conf >= 0.01:
            print(f"DETECTED: {best_text} ({best_conf:.2f})", flush=True)
            return best_text, best_conf, debug_img
            
    except Exception as e:
        print(f"OCR Error: {e}", flush=True)
    
    # Method 2: Contour-based detection (for well-framed plates)
    print("Trying contour-based detection...", flush=True)
    candidates = detect_plate_contours(enhanced)
    for region in candidates[:3]:
        x, y, w, h = region
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        plate_region = extract_plate_region(enhanced, region)
        if plate_region is not None:
            detection_method = "contour"
            try:
                results = reader.readtext(plate_region, contrast_ths=0.3, adjust_contrast=0.5)
                
                for result in results:
                    if len(result) >= 3:
                        text = str(result[1]).strip()
                        conf = float(result[2])
                    else:
                        continue
                    
                    cleaned = ''.join(c for c in text.upper() if c.isalnum())
                    
                    if len(cleaned) >= 4:
                        has_letters = any(c.isalpha() for c in cleaned)
                        has_numbers = any(c.isdigit() for c in cleaned)
                        
                        if (has_letters and has_numbers) or (has_numbers and len(cleaned) >= 4):
                            if conf > best_conf:
                                best_text = cleaned
                                best_conf = conf
                
                if best_text and best_conf >= 0.01:
                    print(f"CONTOUR DETECTED: {best_text} ({best_conf:.2f})", flush=True)
                    return best_text, best_conf, debug_img
                    
            except Exception as e:
                print(f"Contour OCR Error: {e}", flush=True)
    
    # Method 3: Try with different contrast settings
    print("Trying with different OCR settings...", flush=True)
    try:
        results = reader.readtext(
            resized,
            contrast_ths=0.5,
            adjust_contrast=0.8,
            text_threshold=0.4
        )
        
        for result in results:
            if len(result) >= 3:
                text = str(result[1]).strip()
                conf = float(result[2])
                
                cleaned = ''.join(c for c in text.upper() if c.isalnum())
                
                if len(cleaned) >= 4:
                    has_letters = any(c.isalpha() for c in cleaned)
                    has_numbers = any(c.isdigit() for c in cleaned)
                    
                    if (has_letters and has_numbers) or (has_numbers and len(cleaned) >= 5):
                        print(f"  Alt candidate: '{cleaned}' conf: {conf:.2f}", flush=True)
                        if conf > best_conf:
                            best_text = cleaned
                            best_conf = conf
        
        if best_text and best_conf >= 0.01:
            print(f"ALT DETECTED: {best_text} ({best_conf:.2f})", flush=True)
            return best_text, best_conf, debug_img
            
    except Exception as e:
        print(f"Alt OCR Error: {e}", flush=True)
    
    return "No plate detected", 0, debug_img


def detect_from_camera_frame(frame):
    """
    Instant detection for camera frames
    No stability check - shows result immediately
    """
    if frame is None:
        return {"text": "No frame", "confidence": 0}
    
    # Use higher resolution for better OCR (not too small)
    medium = cv2.resize(frame, (640, 480))
    
    # Detect and read - return immediately with higher quality settings
    text, conf, _ = detect_and_read_plate_fast(medium)
    
    return {"text": text, "confidence": round(conf, 2)}


def is_valid_license_plate(text):
    """
    STRICT validation for license plates
    Must have BOTH letters AND numbers, and follow standard plate format
    """
    if not text or len(text) < 4:
        return False
    
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    # License plates MUST have both letters AND numbers
    if not (has_letters and has_numbers):
        return False
    
    # Check for valid plate patterns
    # Standard: 2 letters + 2-4 numbers + 1-3 letters (e.g., MH12AB1234)
    # Or: 2-4 numbers + 2-3 letters (e.g., 1234ABC)
    import re
    patterns = [
        r'^[A-Z]{1,3}[0-9]{2,4}[A-Z]{1,3}$',  # MH12AB1234
        r'^[0-9]{2,4}[A-Z]{1,3}$',             # 1234ABC
        r'^[A-Z]{1,2}[0-9]{1,4}$',            # MH12
        r'^[0-9]{1,4}[A-Z]{1,2}$',            # 1234AB
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    # Also accept if it has 4-10 chars with both letters and numbers
    if 4 <= len(text) <= 10 and has_letters and has_numbers:
        return True
    
    return False


def detect_and_read_plate_fast(image):
    """
    FAST detection for camera frames - with STRICT validation
    Only accepts real license plate patterns
    """
    if image is None:
        return "No image provided", 0, None
    
    debug_img = image.copy()
    reader = get_reader()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize for OCR - use 640x480 for better accuracy
    resized = cv2.resize(gray, (640, 480))
    
    # Apply moiré pattern removal (band-stop filter simulation)
    try:
        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        
        rows, cols = resized.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        img_back = np.uint8(img_back)
        
        resized = cv2.addWeighted(resized, 0.5, img_back, 0.5, 0)
    except:
        pass
    
    try:
        all_candidates = []
        
        # Pass 1: Standard settings
        results = reader.readtext(
            resized,
            contrast_ths=0.3,
            adjust_contrast=0.5,
            text_threshold=0.5,
            width_ths=0.6
        )
        
        for result in results:
            if len(result) >= 3:
                bbox, text, conf = result[0], result[1], result[2]
            elif len(result) >= 2:
                text = result[0] if isinstance(result[0], str) else result[1]
                conf = result[1] if isinstance(result[1], (int, float)) else 0.5
            else:
                continue
            
            text = str(text).strip().upper()
            if not text:
                continue
            
            cleaned = ''.join(c for c in text if c.isalnum())
            
            # STRICT: Only accept valid license plate patterns
            if len(cleaned) >= 4 and is_valid_license_plate(cleaned):
                conf = float(conf) if conf is not None else 0.5
                all_candidates.append((cleaned, conf))
        
        # Pass 2: Higher contrast
        results2 = reader.readtext(
            resized,
            contrast_ths=0.5,
            adjust_contrast=0.8,
            text_threshold=0.4,
            width_ths=0.7
        )
        
        for result in results2:
            if len(result) >= 3:
                text = str(result[1]).strip().upper()
                conf = float(result[2])
                cleaned = ''.join(c for c in text if c.isalnum())
                if len(cleaned) >= 4 and is_valid_license_plate(cleaned):
                    all_candidates.append((cleaned, conf))
        
        # Pass 3: Lower threshold
        results3 = reader.readtext(
            resized,
            contrast_ths=0.1,
            adjust_contrast=0.2,
            text_threshold=0.3,
            width_ths=0.8
        )
        
        for result in results3:
            if len(result) >= 3:
                text = str(result[1]).strip().upper()
                conf = float(result[2])
                cleaned = ''.join(c for c in text if c.isalnum())
                if len(cleaned) >= 4 and is_valid_license_plate(cleaned):
                    all_candidates.append((cleaned, conf))
        
        # Find best candidate with HIGH confidence (at least 30%)
        best_text = ""
        best_conf = 0
        
        for text, conf in all_candidates:
            # Only accept if confidence is at least 30%
            if conf > best_conf and conf >= 0.30:
                best_text = text
                best_conf = conf
        
        if best_text and best_conf >= 0.30:
            print(f"FAST DETECTED: {best_text} ({best_conf:.2f})", flush=True)
            return best_text, best_conf, debug_img
            
    except Exception as e:
        print(f"Fast OCR Error: {e}", flush=True)
    
    return "No plate detected", 0, debug_img


# ============================================
# NEW: MOBILE IMAGE DETECTION FUNCTIONS
# ============================================

def preprocess_for_mobile(image):
    """
    Enhanced preprocessing specifically for MOBILE PHOTOS
    Handles: blur, low resolution, glare, compression artifacts
    """
    if image is None:
        return None
    
    # Make a copy
    img = image.copy()
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Step 1: Upscale low-resolution images (mobile often have small images)
    h, w = gray.shape
    if h < 300 or w < 400:
        scale = max(400/w, 300/h)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled image from {w}x{h}", flush=True)
    
    # Step 2: Reduce noise (mobile images are often noisy)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Step 3: Sharpen the image (mobile photos are often slightly blurred)
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Step 4: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)
    
    # Step 5: Remove moiré patterns (phone screen photos)
    try:
        # Use frequency domain filtering
        f = np.fft.fft2(enhanced)
        fshift = np.fft.fftshift(f)
        
        rows, cols = enhanced.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask to remove periodic patterns
        mask = np.ones((rows, cols), np.uint8)
        r = 40
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
        mask[mask_area] = 0
        
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        img_back = np.uint8(img_back)
        
        # Blend
        enhanced = cv2.addWeighted(enhanced, 0.6, img_back, 0.4, 0)
    except:
        pass
    
    # Step 6: Apply adaptive thresholding for better text extraction
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return gray, enhanced, thresh


def detect_mobile_plate(image):
    """
    INSTANT detection optimized for MOBILE PHOTOS
    Returns result immediately without stability check
    With STRICT validation - only accepts real license plates
    """
    if image is None:
        return "No image provided", 0, None
    
    print("Using MOBILE detection mode...", flush=True)
    reader = get_reader()
    
    # Preprocess for mobile
    gray, enhanced, thresh = preprocess_for_mobile(image)
    
    # Try multiple image versions
    image_versions = [
        ("enhanced", enhanced),
        ("thresh", thresh),
        ("original", gray),
    ]
    
    best_text = ""
    best_conf = 0
    
    for version_name, img in image_versions:
        print(f"  Trying {version_name}...", flush=True)
        
        # Resize for better OCR
        if img is not None:
            resized = cv2.resize(img, (800, 600))
            
            # Try multiple OCR settings optimized for mobile
            settings = [
                {"contrast_ths": 0.1, "adjust_contrast": 0.3, "text_threshold": 0.3, "width_ths": 0.8},
                {"contrast_ths": 0.2, "adjust_contrast": 0.5, "text_threshold": 0.5, "width_ths": 0.6},
                {"contrast_ths": 0.4, "adjust_contrast": 0.7, "text_threshold": 0.4, "width_ths": 0.7},
                {"contrast_ths": 0.15, "adjust_contrast": 0.4, "text_threshold": 0.25, "width_ths": 0.9},
            ]
            
            for i, params in enumerate(settings):
                try:
                    results = reader.readtext(resized, **params)
                    
                    for result in results:
                        if len(result) >= 3:
                            text = str(result[1]).strip().upper()
                            conf = float(result[2])
                        elif len(result) >= 2:
                            text = str(result[0]).strip().upper() if isinstance(result[0], str) else str(result[1]).strip().upper()
                            conf = float(result[1]) if isinstance(result[1], (int, float)) else 0.5
                        else:
                            continue
                        
                        if not text:
                            continue
                        
                        # Clean text - keep alphanumeric
                        cleaned = ''.join(c for c in text if c.isalnum())
                        
                        # STRICT: Must be valid license plate with BOTH letters and numbers
                        if len(cleaned) >= 4 and is_valid_license_plate(cleaned):
                            if conf > best_conf:
                                best_text = cleaned
                                best_conf = conf
                                print(f"    Mobile candidate: '{cleaned}' conf: {conf:.2f}", flush=True)
                except Exception as e:
                    continue
    
    # Require at least 30% confidence for mobile
    if best_text and best_conf >= 0.30:
        print(f"MOBILE DETECTED: {best_text} ({best_conf:.2f})", flush=True)
        return best_text, best_conf, image.copy()
    
    return "No plate detected", 0, None


def detect_instant(image):
    """
    INSTANT detection - returns immediately with best result
    Combines mobile-optimized and fast detection
    """
    if image is None:
        return "No plate detected", 0
    
    # Try mobile detection first (more thorough)
    text, conf, _ = detect_mobile_plate(image)
    
    if text and text != "No plate detected" and conf > 0.03:
        return text, conf
    
    # Fallback to fast detection
    text, conf, _ = detect_and_read_plate_fast(image)
    
    if text and text != "No plate detected" and conf > 0.03:
        return text, conf
    
    return text if text else "No plate detected", conf

