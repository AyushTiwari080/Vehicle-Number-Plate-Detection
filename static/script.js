// Vehicle Number Plate Detection - Frontend JavaScript
// Handles video streaming, controls, and real-time updates

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoStream = document.getElementById('videoStream');
    const toggleBtn = document.getElementById('toggleBtn');
    const captureBtn = document.getElementById('captureBtn');
    const plateText = document.getElementById('plateText');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const statusDot = document.querySelector('.status-dot');

    let isStreaming = true;
    let detectionEnabled = true;

    // Initialize video stream
    initVideoStream();

    function initVideoStream() {
        // Show loading spinner initially
        if (loadingSpinner) {
            loadingSpinner.classList.add('visible');
        }

        // Hide loading spinner after video loads (with fallback timeout)
        const hideSpinner = () => {
            if (loadingSpinner) {
                loadingSpinner.classList.remove('visible');
            }
            updateStatus(true);
        };

        // For image/video streams, use setTimeout as fallback
        setTimeout(hideSpinner, 3000);
        
        // Also try onload if available
        videoStream.onload = hideSpinner;
        
        // Handle video stream errors
        videoStream.onerror = () => {
            showError('Video stream failed. Please refresh the page.');
        };
    }

    // Toggle detection button
    if (toggleBtn) {
        toggleBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/toggle_detection', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const data = await response.json();

                if (data.detection_active) {
                    toggleBtn.classList.add('active');
                    toggleBtn.querySelector('i').className = 'fas fa-pause';
                    toggleBtn.querySelector('span').textContent = 'Pause';
                    detectionEnabled = true;
                } else {
                    toggleBtn.classList.remove('active');
                    toggleBtn.querySelector('i').className = 'fas fa-play';
                    toggleBtn.querySelector('span').textContent = 'Resume';
                    detectionEnabled = false;
                }
            } catch (error) {
                console.error('Toggle error:', error);
            }
        });
    }

    // Quick Detect button - instant detection from sample image
    const quickDetectBtn = document.getElementById('quickDetectBtn');
    if (quickDetectBtn) {
        quickDetectBtn.addEventListener('click', async () => {
            try {
                quickDetectBtn.querySelector('i').classList.add('fa-spin');
                quickDetectBtn.querySelector('span').textContent = 'Detecting...';
                
                const response = await fetch('/detect_sample');
                const data = await response.json();
                
                if (data.text && data.text !== "Sample image not found") {
                    updatePlateDisplay(data.text, data.confidence || 0);
                    showNotification(`Detected: ${data.text}`);
                } else {
                    showNotification('No plate detected in sample');
                }
            } catch (error) {
                console.error('Quick detect error:', error);
                showNotification('Detection failed');
            } finally {
                quickDetectBtn.querySelector('i').classList.remove('fa-spin');
                quickDetectBtn.querySelector('span').textContent = 'Quick Detect';
            }
        });
    }

    // Capture button
    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            captureFrame();
        });
    }

    // Upload button - handle image upload
    const uploadInput = document.getElementById('uploadInput');
    if (uploadInput) {
        uploadInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                showNotification('Processing image...');
                
                const formData = new FormData();
                formData.append('image', file);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.text && data.text !== "No plate detected") {
                    updatePlateDisplay(data.text, data.confidence || 0);
                    showNotification(`Detected: ${data.text}`);
                } else {
                    showNotification('No plate detected in image');
                }
            } catch (error) {
                console.error('Upload error:', error);
                showNotification('Upload failed');
            }
            
            // Clear input for next upload
            uploadInput.value = '';
        });
    }

    function captureFrame() {
        // Create a canvas to capture the current frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Set canvas size to match video
        canvas.width = videoStream.naturalWidth || 640;
        canvas.height = videoStream.naturalHeight || 480;

        // Draw current video frame to canvas
        ctx.drawImage(videoStream, 0, 0);

        // Convert to image and download
        const link = document.createElement('a');
        link.download = `plate_capture_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();

        // Visual feedback
        showNotification('Frame captured!');
    }

    // Poll for detection results - FAST polling for instant detection
    setInterval(async () => {
        if (!detectionEnabled) return;

        try {
            // Fetch plate text from /status endpoint
            const statusResponse = await fetch('/status');
            const statusData = await statusResponse.json();
            
            // Fetch confidence from /result endpoint
            const resultResponse = await fetch('/result');
            const resultData = await resultResponse.json();

            if (statusData.current_plate && statusData.current_plate !== "No plate detected") {
                updatePlateDisplay(statusData.current_plate, resultData.confidence || 0);
            }
        } catch (error) {
            console.error('Status fetch error:', error);
        }
    }, 500);  // 500ms for instant detection

    function updatePlateDisplay(text, confidence) {
        if (plateText) {
            plateText.textContent = text;
        }

        // Update confidence with actual backend confidence
        if (confidenceFill && confidenceText) {
            const confidencePercent = Math.round(confidence * 100);
            confidenceFill.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `Confidence: ${confidencePercent}%`;
        }
    }

    function updateStatus(active) {
        if (statusDot) {
            if (active) {
                statusDot.classList.add('active');
            } else {
                statusDot.classList.remove('active');
            }
        }
    }

    function showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        // Add animation keyframes
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    function showError(message) {
        if (loadingSpinner) {
            loadingSpinner.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i>
                <span>${message}</span>
            `;
            loadingSpinner.classList.add('visible');
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            toggleBtn.click();
        } else if (e.code === 'KeyC') {
            captureBtn.click();
        }
    });

    // Touch support for mobile
    let touchStartX = 0;
    let touchEndX = 0;

    videoStream.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });

    videoStream.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });

    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe left - next action
                console.log('Swipe left detected');
            } else {
                // Swipe right - previous action
                console.log('Swipe right detected');
            }
        }
    }

    console.log('Vehicle Number Plate Detection App initialized');
    console.log('Keyboard shortcuts: Space = Toggle, C = Capture');
});
