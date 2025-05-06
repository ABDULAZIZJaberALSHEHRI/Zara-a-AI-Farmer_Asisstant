"""
Entry point for the Smart Farming Assistant application.
"""
import os
from app import build_app
from modules.knowledge_base import prepare_chroma_from_local_pdfs
from config import OPENAI_API_KEY, BACKGROUND_IMAGE_PATH, LOGO_PATH

def main():
    """
    Main function to start the application.
    """
    # Check configuration
    print(f"API Key status: {'Found in environment' if OPENAI_API_KEY else 'Not found in environment'}")
    print(f"Using local background image: {BACKGROUND_IMAGE_PATH}")
    print(f"Using logo image: {LOGO_PATH}")
    print("Note: Chroma from langchain is deprecated. Consider updating to langchain-chroma in future versions.")
    
    # Prepare knowledge base
    prepare_chroma_from_local_pdfs()
    
    # Build and launch the app
    app = build_app()
    app.launch()

if __name__ == "__main__":
    main()