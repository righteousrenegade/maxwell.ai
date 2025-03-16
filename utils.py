import os
import logging
import sys
from tqdm import tqdm
import requests
import zipfile
import io

def setup_logger():
    """Set up and return a logger for the application."""
    logger = logging.getLogger("maxwell")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()

def download_file(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_and_extract_zip(url, destination_dir):
    """Download a zip file and extract it."""
    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    z = zipfile.ZipFile(io.BytesIO(response.content))
    logger.info(f"Extracting to {destination_dir}...")
    z.extractall(destination_dir)

def download_models(offline_mode=False):
    """Download required models if they don't exist."""
    # Create models directory
    models_dir = os.path.join(os.path.expanduser("~"), ".maxwell")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download Vosk model for offline recognition if needed
    if offline_mode:
        vosk_model_dir = os.path.join(models_dir, "vosk_model")
        if not os.path.exists(vosk_model_dir):
            logger.info("Downloading Vosk model for offline speech recognition...")
            os.makedirs(vosk_model_dir, exist_ok=True)
            # Small English model
            vosk_model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            download_and_extract_zip(vosk_model_url, vosk_model_dir)
            logger.info("Vosk model downloaded and extracted.")
    
    # Kokoro TTS models will be downloaded automatically when first used
    logger.info("Kokoro TTS models will be downloaded automatically when needed.") 