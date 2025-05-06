"""
Audio transcription functionality.
"""
from openai import OpenAI
from config import OPENAI_API_KEY, WHISPER_MODEL

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_path):
    """
    Transcribe audio to text.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text or error message
    """
    if not audio_path:
        return ""
    
    if not OPENAI_API_KEY:
        return "⚠️ No OpenAI API key available. Audio transcription is disabled."
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="text"
            )
        return transcript
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"