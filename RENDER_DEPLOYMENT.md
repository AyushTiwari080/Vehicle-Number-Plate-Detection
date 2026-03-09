# Deployment Guide for Render

This guide provides step-by-step instructions to deploy the Vehicle Number Plate Detection application to Render.

## Running Locally (Development)

### Option 1: Using Python directly

1. **Navigate to the project directory:**
```bash
cd "c:/Users/MAHAKAAL/OneDrive/Desktop/New folder/Vehicle-Number-Plate-Detection"
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
python app.py
```

5. **Open in browser:**
Navigate to `http://localhost:5000`

### Option 2: Using Gunicorn

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Deploying to Render (Production)

## Prerequisites

- A GitHub repository with your code
- A Render account (free tier works)

## Step 1: Prepare Your Code for Render

### 1.1 Create a `runtime.txt` file
Specify the Python version:

```
python-3.11.0
```

### 1.2 Create a `build.sh` file (optional but recommended)
Create a build script to handle model caching:

```bash
#!/bin/bash
# Build script for Render
# This helps with model caching and setup

echo "Installing dependencies..."
pip install -r requirements.txt

# Create a .cache directory for EasyOCR models
mkdir -p ~/.EasyOCR
echo "Build complete!"
```

### 1.3 Update requirements.txt
Ensure you have all necessary dependencies. Your current requirements.txt is compatible:

```
Flask==3.0.0
opencv-python-headless==4.8.1.78
numpy<2
imutils==0.5.4
easyocr==1.7.0
torch==2.1.0
torchvision==0.16.0
Pillow==10.1.0
python-dotenv==1.0.0
gunicorn==21.2.0
Flask-CORS==4.0.0
```

### 1.4 Update app.py for Cloud Deployment
The app currently tries to access camera which won't work on Render. You need to modify it to use sample image mode by default.

Create a new file or update `app.py` with the following changes:

**Important**: The key issue is that `cv2.VideoCapture(0)` won't work on Render since there's no camera connected. The app needs to detect if it's running on Render and use sample image mode.

## Step 2: Create Render Configuration Files

### 2.1 Update Procfile
Your existing Procfile is correct:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### 2.2 Create render.yaml (for automatic deployment)
```yaml
services:
  - type: web
    name: vehicle-plate-detection
    env: python
    region: Oregon
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

## Step 3: Deploy to Render

### Option A: Manual Deployment

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Prepare for Render deployment"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Create a new Web Service on Render**
   - Go to https://dashboard.render.com
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Configure:
     - Name: vehicle-plate-detection
     - Region: Oregon (or closest to you)
     - Branch: main
     - Runtime: Python
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Click "Create Web Service"

3. **Wait for deployment**
   - First deployment takes 5-10 minutes (downloading EasyOCR models)
   - Render will show build logs

### Option B: Using render.yaml

1. Push the render.yaml file to your GitHub repository
2. Go to Render Dashboard
3. Click "New +" and select "Blueprint"
4. Connect your repository with render.yaml
5. Click "Apply"

## Step 4: Verify Deployment

After deployment, your app should:
1. Load without camera errors
2. Display the sample image (81suTTBxfuL.jpg)
3. Allow image upload for plate detection

## Known Limitations on Render

1. **No real-time camera**: The camera feed won't work on Render. The app will automatically fallback to sample image mode.
2. **Cold starts**: First request after inactivity may take longer
3. **Model loading**: EasyOCR models are downloaded on first run

## Troubleshooting

### Build Failures
- Ensure `requirements.txt` has correct versions
- Check that all files are committed to Git

### Runtime Errors
- Verify the sample image exists (81suTTBxfuL.jpg)
- Check Render logs for errors

### Slow Performance
- EasyOCR models are cached after first load
- Consider upgrading to a paid tier for better performance

## Environment Variables

No additional environment variables are required. The app uses:
- `$PORT` (automatically provided by Render)

## Support

For issues, check:
1. Render dashboard logs
2. Build logs during deployment
3. Application runtime logs

