"""
Simple test script to check camera and OCR
"""
import cv2
import easyocr

print("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=False, verbose=True)
print("OCR Loaded!")

print("\nOpening camera...")
cap = cv2.VideoCapture(0)

print("Taking a photo...")
ret, frame = cap.read()

if ret:
    print("Photo taken!")
    
    # Save the frame
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved to test_frame.jpg")
    
    # Try OCR
    print("Running OCR...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    
    print(f"\nFound {len(results)} text regions:")
    for bbox, text, conf in results:
        print(f"  - '{text}' (confidence: {conf:.3f})")
else:
    print("Failed to capture frame")

cap.release()

