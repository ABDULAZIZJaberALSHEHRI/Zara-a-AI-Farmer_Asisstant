
# 🌱 ZARA'A – Smart Farming Assistant

ZARA'A is a multimodal AI assistant designed to support farmers by analyzing plant health, responding to queries via voice, image, or text input, and retrieving useful agricultural knowledge.

## 🧠 Features

- **Multimodal Input**: Supports image, audio, and text queries.
- **Plant Disease Detection**: Upload an image of a plant to identify potential diseases.
- **Farming Q&A**: Ask questions using text, audio, or images.
- **Retrieval-Augmented Generation (RAG)**: Intelligent responses backed by custom document retrieval.
- **Modern UI**: Stylish frontend with integrated CSS for a smooth experience.

## 🗂 Project Structure

```
zaraa v4/
├── app.py                  # Main app entrypoint
├── main.py                 # Starts the application logic
├── build_chroma_once.py   # Initializes the ChromaDB vector store
├── config.py               # Configuration and API keys
├── style.css               # Custom UI styling
├── requirement.txt         # Python dependencies
├── .env                    # Environment variables (API keys etc.)
├── modules/                # Core modules
│   ├── agent.py            # LangChain agent logic
│   ├── audio.py            # Audio processing with Whisper
│   ├── chat.py             # Chat flow and orchestration
│   ├── images_detector.py  # Plant disease detection
│   ├── retrival.py         # Document retrieval logic
│   └── utils.py            # Utility functions
└── ...
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourname/zaraa-v4.git
cd zaraa-v4
```

### 2. Set Up Environment
```bash
python -m venv farming_env
source farming_env/bin/activate  # On Windows: farming_env\Scripts\activate
pip install -r requirement.txt
```

### 3. Add Environment Variables
Create a `.env` file and include your API keys:
```
OPENAI_API_KEY=your-key
```

### 4. Initialize Vector Store
```bash
python build_chroma_once.py
```

### 5. Run the App
```bash
python main.py
```

## 📽 Demo Video

Watch the full walkthrough of the project deployment:

[📹 View on Google Drive](https://drive.google.com/drive/folders/1stX8JwcTGDCcLD9PJBk9pbLYKnnzx1Ri?usp=sharing)  

## 🛠 Tech Stack

- **Python** (LangChain, OpenAI, ChromaDB, Whisper)
- **FastAPI / Gradio** (Frontend)
- **Custom CSS** (User Interface)
- **Pydub + ffmpeg** (Audio processing)

## 📦 Dependencies

Install required packages:
```bash
pip install -r requirement.txt
```

## 🧪 Example Use Cases

- Ask “How to grow tomatoes in hot weather?” via voice or text.
- Upload a photo of a diseased leaf to get an instant diagnosis.
- Talk to Zaraa about pest control methods.

## 👨‍🌾 Authors

- Project led by: Abdulaziz Alshehri
- AI Bootcamp Final Project
