"""
Configuration settings for the Smart Farming Assistant.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key Management
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_PROJECT = "zaraa-farmer-project"

# Model configurations
MODEL_BEAN_CLASSIFIER = "nateraw/vit-base-beans"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GPT_CHAT_MODEL = "gpt-3.5-turbo"
GPT_CHAT_MODEL_LARGE = "gpt-3.5-turbo-16k"
WHISPER_MODEL = "whisper-1"

# Paths
DATA_DIR = "data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
BACKGROUND_IMAGE_PATH = os.path.join(DATA_DIR, "Untitled desig.png")
LOGO_PATH = os.path.join(DATA_DIR, "logo.png")

# Vector store
CHROMA_COLLECTION_NAME = "plant_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Device settings
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# Debug settings
DEBUG = os.environ.get("DEBUG", "0") == "1"