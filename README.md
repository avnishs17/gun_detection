# Gun Detection MLOps Project

A computer vision project for gun detection using Faster R-CNN, implemented with MLOps best practices including DVC for data versioning, Google Cloud Storage for remote storage, and FastAPI for model serving.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [DVC and Version Control Setup](#dvc-and-version-control-setup)
- [Google Cloud Platform Setup](#google-cloud-platform-setup)
- [Running the Application](#running-the-application)
- [API Testing](#api-testing)

## Prerequisites

- Python 3.11.13
- Conda or virtual environment manager
- Git
- Google Cloud Platform account
- Kaggle account (for data access)

## Environment Setup

1. **Create and activate virtual environment:**
   ```bash
   conda create -p gun_detection_env python==3.11.13
   conda activate gun_detection_env/
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

## Data Preparation

1. **Clear Kaggle cache (if needed):**
   ```bash
   # Clear the kagglehub cache to ensure fresh download
   # Navigate to: C:/Users/your_username/.cache/kagglehub
   # Delete the kagglehub folder if it exists
   ```

2. **Download data from Kaggle:**
   ```bash
   python src/data_ingestion.py
   ```

## Model Training

Run the training script to train the Faster R-CNN model:
```bash
python src/model_training.py
```

## DVC and Version Control Setup

1. **Initialize Git repository:**
   ```bash
   git init
   git branch -M main
   git remote add origin https://github.com/your-username/your-repo.git
   ```

2. **Initialize DVC:**
   ```bash
   dvc init --no-scm
   ```

3. **Create DVC pipeline:**
   - Configure `dvc.yaml` with your pipeline commands
   - Run the pipeline:
   ```bash
   dvc repro
   ```

## Google Cloud Platform Setup

**Important:** Complete this setup before proceeding with data versioning and remote storage.

### Step 1: Create GCS Bucket

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Cloud Storage → Buckets
2. Click "Create Bucket"
3. Provide a unique bucket name (e.g., `your-gun-detection-bucket`)
4. In "Choose how to control access to objects":
   - **Uncheck** "Enforce public access prevention on this bucket"
5. Complete bucket creation

### Step 2: Create Service Account

1. Navigate to IAM & Admin → Service Accounts
2. Click "Create Service Account"
3. Provide a name (e.g., `gun-detection-service-account`)
4. Assign the following roles:
   - **Owner**
   - **Storage Object Admin**
   - **Storage Object Viewer**
5. Click "Continue" → "Done"

### Step 3: Generate Service Account Key

1. Find your service account in the list
2. Click the three dots (⋮) → "Manage Keys"
3. Click "Add Key" → "Create New Key"
4. Select "JSON" format and download the key file
5. **Important:** Store this file securely

### Step 4: Configure Bucket Permissions

1. Return to Cloud Storage → Buckets
2. Find your bucket and click the three dots (⋮)
3. Click "Edit Access"
4. Add your service account with roles:
   - Storage Object Admin
   - Storage Object Viewer
5. Save changes

### Step 5: Configure DVC Remote Storage

1. **Add DVC remote:**
   ```bash
   dvc remote add -d myremote gs://your-bucket-name/
   ```

2. **Set up authentication:**
   ```bash
   $env:GOOGLE_APPLICATION_CREDENTIALS = "path/to/your-service-account-key.json"
   ```

3. **Push data to remote storage:**
   ```bash
   dvc push
   ```

4. **Commit and push to Git:**
   ```bash
   # Ensure .gitignore is properly configured
   git add .
   git commit -m "Initial commit with DVC setup"
   git push -u origin main
   ```

## Running the Application

1. **Ensure model exists:**
   - Verify that `artifacts/models/fasterrcnn.pth` exists

2. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the application:**
   - API will be available at: `http://127.0.0.1:8000`
   - Interactive docs at: `http://127.0.0.1:8000/docs`

## API Testing

### Using Postman

1. **Create new request:**
   - Method: `POST`
   - URL: `http://127.0.0.1:8000/predict/`

2. **Configure request body:**
   - Select "Body" → "form-data"
   - Key: `file` (change type from "Text" to "File")
   - Value: Select an image file from `artifacts/raw/Images/`

3. **Send request:**
   - Click "Send" to get the prediction results


## Project Structure

```
├── artifacts/          # Data and model artifacts
│   ├── models/        # Trained model files
│   └── raw/           # Raw data (Images, Labels)
├── config/            # Configuration files
├── logs/              # Application logs
├── notebook/          # Jupyter notebooks
├── src/              # Source code modules
├── tensorboard_logs/ # TensorBoard training logs and metrics
├── dvc.yaml          # DVC pipeline configuration
├── main.py           # FastAPI application
├── setup.py          # Package setup and installation
└── requirements.txt  # Python dependencies
```

## Notes

- Ensure all dependencies are installed before running any scripts
- Keep your GCP service account key secure and never commit it to version control
- Configure `.gitignore` appropriately to exclude sensitive files and large data files
- Use DVC for tracking large files and datasets