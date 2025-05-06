"""
Plant disease detection functionality.
"""
import torch
import numpy as np
from PIL import Image
from modules.model_loader import load_image_classification_model, load_plant_dataset, load_embeddings_model
from modules.fruit_classifier import classify_fruit_or_vegetable
import matplotlib.pyplot as plt

# Load models and dataset
processor, model, class_labels = load_image_classification_model()
descriptions, labels, description_embeddings = load_plant_dataset()
embedder = load_embeddings_model()

def plot_top_predictions(predictions):
    labels = [label.replace('_', ' ').title() for label, _ in predictions]
    scores = [score for _, score in predictions]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.bar(labels, scores, color=['#34D399', '#10B981', '#059669'])

    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence", fontsize=12, color='white')
    ax.set_title("Top Predictions", fontsize=14, color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    return fig


def predict_image(image):
    """
    Predict plant disease from an image.
    
    Args:
        image: Path to the image file or image object
        
    Returns:
        tuple: (display_img, prediction, top_predictions, description, treatment)
    """
    if image is None:
        return None, "No image uploaded", "", "", ""
    
    try:
        # Load and preprocess image
        image_pil = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        display_img = image_pil.copy()
        
        # Prepare inputs for the model
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_disease = class_labels[predicted_class_idx]
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        
        # Get top 3 predictions for display
        top_idxs = torch.topk(logits, k=3).indices[0].tolist()
        top_predictions = [(class_labels[idx], torch.softmax(logits, dim=1)[0][idx].item()) for idx in top_idxs]
        top_predictions_plot = plot_top_predictions(top_predictions)

        
        # Get matching description
        query_embedding = embedder.encode(predicted_disease, normalize_embeddings=True)
        similarities = np.dot(description_embeddings, query_embedding)
        top_match_idx = np.argmax(similarities)
        matched_label = labels[top_match_idx]
        matched_description = descriptions[top_match_idx]
        
        # Generate treatment recommendations
        treatment_recommendations = generate_treatment_tips(matched_label)
        
        return (
            f"**Prediction: {predicted_disease.replace('_', ' ').title()}** ({confidence:.1%})", 
            top_predictions_plot, 
            matched_description, 
            treatment_recommendations
        )

    
    except Exception as e:
        return None, f"⚠️ Error processing image: {str(e)}", "", "", ""

def generate_treatment_tips(disease_name):
    """
    Generate treatment recommendations based on the disease.
    
    Args:
        disease_name: Name of the identified disease
        
    Returns:
        str: Treatment recommendations
    """
    common_tips = {
        "angular leaf spot": "• Remove infected leaves\n• Apply copper-based fungicides\n• Ensure proper spacing between plants\n• Avoid overhead irrigation",
        "bean rust": "• Apply fungicides at first sign of infection\n• Plant resistant varieties\n• Remove infected plant debris\n• Rotate crops",
        "healthy": "• Continue regular monitoring\n• Maintain balanced fertilization\n• Water appropriately\n• Practice crop rotation",
    }
    
    # Default recommendations if specific disease not found
    default_tips = "• Remove infected plant material\n• Consider appropriate fungicides\n• Ensure good air circulation\n• Avoid overhead watering\n• Practice crop rotation"
    
    # Check for partial matches if not exact
    if disease_name.lower() not in common_tips:
        for key in common_tips:
            if key in disease_name.lower():
                return common_tips[key]
    
    return common_tips.get(disease_name.lower(), default_tips)

def analyze_uploaded_plant_image(image_path):
    if not image_path:
        return None

    try:
        predicted_label = classify_fruit_or_vegetable(image_path)
        return predicted_label
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
