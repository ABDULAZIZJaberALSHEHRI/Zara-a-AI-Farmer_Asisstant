"""
Model loading functions for the Smart Farming Assistant.
"""
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from config import MODEL_BEAN_CLASSIFIER, EMBEDDING_MODEL, DEVICE

def load_image_classification_model():
    """
    Load the plant disease classification model.
    
    Returns:
        tuple: (processor, model, class_labels) tuple
    """
    # Load image processor
    processor = AutoImageProcessor.from_pretrained(MODEL_BEAN_CLASSIFIER)
    
    # Load classification model
    model = AutoModelForImageClassification.from_pretrained(MODEL_BEAN_CLASSIFIER)
    model = model.to(DEVICE)
    
    # Get class labels
    class_labels = model.config.id2label
    
    return processor, model, class_labels

def load_embeddings_model():
    """
    Load the sentence embeddings model.
    
    Returns:
        SentenceTransformer: The embeddings model
    """
    return SentenceTransformer(EMBEDDING_MODEL)

def load_plant_dataset():
    """
    Load the plant disease dataset.
    
    Returns:
        tuple: (descriptions, labels, embeddings) tuple
    """
    # Load dataset
    dataset = load_dataset("ipranavks/plant-disease-datasetog")
    
    # Extract descriptions and labels
    descriptions = [sample['description'] for sample in dataset['train']]
    labels = [sample['label'] for sample in dataset['train']]
    
    # Compute embeddings
    embedder = load_embeddings_model()
    description_embeddings = embedder.encode(descriptions, normalize_embeddings=True)
    
    return descriptions, labels, description_embeddings