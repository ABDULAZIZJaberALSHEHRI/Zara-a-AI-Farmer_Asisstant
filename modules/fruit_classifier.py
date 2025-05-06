from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor once
processor = AutoImageProcessor.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
model.eval()

class_labels = model.config.id2label

def classify_fruit_or_vegetable(image_path):
    """
    Classify image into fruit/vegetable name.

    Args:
        image_path: Path to image

    Returns:
        str: Predicted label (e.g., Apple, Carrot, etc.)
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = class_labels[predicted_class_idx]

    return predicted_label
