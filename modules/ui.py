"""
UI components and styling for the Smart Farming Assistant.
"""
import base64
import os
from config import BACKGROUND_IMAGE_PATH, LOGO_PATH

def get_local_image_css(image_path):
    """
    Generate CSS for background image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: CSS code for background image
    """
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            return f"""
            .gradio-container {{
                background-image: url('data:image/jpeg;base64,{encoded_string}');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                position: relative;
                color: white !important;
            }}
            """
    except Exception as e:
        print(f"Error loading background image: {str(e)}")
        # Fallback to a solid color background if image cannot be loaded
        return """
        .gradio-container {
            background-color: #0f172a;
            position: relative;
            color: white !important;
        }
        """

def get_image_data_url(image_path):
    """
    Convert image to data URL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Data URL or None
    """
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            ext = os.path.splitext(image_path)[1][1:]  # Get extension without dot
            return f"data:image/{ext};base64,{encoded_string}"
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def get_logo_html():
    """
    Generate HTML for logo.
    
    Returns:
        str: HTML code for logo
    """
    try:
        logo_data_url = get_image_data_url(LOGO_PATH)
        # Define logo HTML with fallback to text if image loading fails
        if logo_data_url:
            logo_html = f"""
            <div class="logo-container">
                <img src="{logo_data_url}" alt="Smart Farming Assistant Logo" class="app-logo">
                <h1>Smart Farming Assistant</h1>
            </div>
            """
        else:
            # Fallback to just text with icon if image can't be loaded
            logo_html = """
            <div class="logo-container">
                <h1>Smart Farming Assistant</h1>
            </div>
            """
    except Exception as e:
        print(f"Error preparing logo: {str(e)}")
        # Fallback if something goes wrong with logo preparation
        logo_html = """
        <div class="logo-container">
            <h1>🌽 Smart Farming Assistant</h1>
        </div>
        """
    
    return logo_html

def load_css_file(filepath):
    """
    Load CSS from a file.
    
    Args:
        filepath: Path to the CSS file
        
    Returns:
        str: CSS code from file
    """
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading CSS file: {str(e)}")
        return ""

def get_custom_css():
    try:
        bg_css = get_local_image_css(BACKGROUND_IMAGE_PATH)
        css_file_path = os.path.join(os.path.dirname(__file__), "../style.css")
        with open(css_file_path, "r", encoding="utf-8") as f:
            base_css = f.read()
        return bg_css + base_css
    except Exception as e:
        print(f"Error loading styles, using fallback: {str(e)}")
        return """
        .gradio-container {
            background-color: #0f172a;
            color: white !important;
        }
        """
