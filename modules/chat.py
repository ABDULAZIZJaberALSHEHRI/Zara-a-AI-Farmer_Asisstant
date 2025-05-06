"""
Chat functionality for the Smart Farming Assistant.
"""
from langsmith import traceable
from modules.agent import initialize_farming_agent, initialize_qa_chain
from config import OPENAI_API_KEY

# Initialize the farming agent and QA chain
farming_agent = None
qa_chain = None

# Create a local conversation context
conversation_context = {
    "last_topic": None,
    "previous_queries": [],
    "previous_responses": []
}

# Maximum history to keep
MAX_CONTEXT_ITEMS = 5

def clean_source_text(source):
    """
    Clean source text by removing .pdf extensions and replacing underscores.
    
    Args:
        source: Raw source text
        
    Returns:
        str: Cleaned source text
    """
    # Split by page if present
    if ", Page" in source:
        name_part, page_part = source.rsplit(", Page", 1)
    else:
        name_part, page_part = source, ""
    
    # Clean the name part
    name_clean = name_part.replace(".pdf", "").replace("_", " ").replace("-", " ").strip()
    
    # Reconstruct with page if present
    if page_part:
        return f"{name_clean}, Page {page_part.strip()}"
    return name_clean

if OPENAI_API_KEY:
    farming_agent = initialize_farming_agent(OPENAI_API_KEY)
    agent_knowledge_base = initialize_farming_agent(OPENAI_API_KEY)
    qa_chain = initialize_qa_chain(OPENAI_API_KEY)

def identify_topic(message):
    """
    Identify the main topic from a message with support for multi-word topics.
    
    Args:
        message: The user message
        
    Returns:
        str: The identified topic
    """
    # List of keywords to ignore
    ignore_words = ["how", "can", "i", "grow", "about", "what", "is", "tell", "me", "more", "explain", 
                   "the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their"]
    
    # Extract words and clean up
    words = message.lower().replace("?", "").replace(".", "").replace("!", "").split()
    
    # Try to find disease or plant names (typically 2-3 word phrases)
    # Common patterns for diseases: "X leaf spot", "X blight", "X rot", etc.
    disease_patterns = ["leaf", "spot", "blight", "rot", "rust", "mildew", "wilt", "mold", "canker", "disease"]
    
    # Look for disease patterns
    for i, word in enumerate(words):
        if word in disease_patterns and i > 0:
            # Found potential disease name, try to construct a multi-word topic
            start = max(0, i-2)  # Go back up to 2 words
            potential_topic = " ".join(words[start:i+1])
            return potential_topic
    
    # If no disease pattern was found, use the original logic but try to get meaningful phrases
    # Filter out common words
    topic_words = [word for word in words if word not in ignore_words and len(word) > 2]
    
    if len(topic_words) >= 2:
        # Return the first two meaningful words as a potential topic
        return " ".join(topic_words[:2])
    elif topic_words:
        return topic_words[0]
    
    return None

@traceable(name="SmartFarmingChat", tags=["chat", "agent", "qa"])
def agent_chatbot_response(user_message, history):
    """
    Generate chatbot response using the farming agent.
    
    Args:
        user_message: User's message
        history: Chat history
        
    Returns:
        list: Updated chat history
    """
    global conversation_context
    
    if not user_message:
        return history
    
    # Check if API keys are available
    if not farming_agent or not qa_chain:
        response = "⚠️ No OpenAI API key available in environment variables. Chat features are disabled."
        history.append((user_message, response))
        return history
    
    # Check if this is a follow-up question
    follow_up_phrases = ["tell me more", "explain more", "additional information", "continue", "elaborate", 
                         "go on", "what else", "and", "more details", "how to treat", "treatment", 
                         "how to fix", "how to cure", "is there a way", "can i", "should i",
                         "this disease", "the disease", "this problem", "the problem"]
    
    # Enhanced follow-up detection
    is_follow_up = (any(phrase in user_message.lower() for phrase in follow_up_phrases) or 
                   len(user_message.split()) <= 5 or
                   user_message.lower().startswith("how") or
                   "this" in user_message.lower() or 
                   "it" in user_message.lower())
    
    # Process the user message
    try:
        # REPLACE THIS BLOCK with the improved follow-up handling block
        # If it's a follow-up and we have previous context
        if is_follow_up and conversation_context["last_topic"]:
            # Get previous query for context
            prev_query = ""
            if conversation_context["previous_queries"]:
                prev_query = conversation_context["previous_queries"][-1]
            
            # Create a more specific query based on the follow-up type
            if "treat" in user_message.lower() or "cure" in user_message.lower() or "fix" in user_message.lower():
                topic_query = f"How to treat {conversation_context['last_topic']} disease"
            elif "prevent" in user_message.lower():
                topic_query = f"How to prevent {conversation_context['last_topic']} disease"
            elif "cause" in user_message.lower():
                topic_query = f"What causes {conversation_context['last_topic']} disease"
            elif "way" in user_message.lower() and "treat" in user_message.lower():
                topic_query = f"Treatment methods for {conversation_context['last_topic']}"
            else:
                topic_query = f"{user_message} about {conversation_context['last_topic']}"
            
            # Log for debugging
            print(f"Follow-up detected. Original: '{user_message}', Using topic query: '{topic_query}'")
            
            try:
                # Use the knowledge base with the enhanced topic query
                qa_result = qa_chain(topic_query)
                response = qa_result["result"]
                
                # Add source attribution
                source_docs = qa_result.get("source_documents", [])
                if source_docs:
                    sources = []
                    for doc in source_docs:
                        if hasattr(doc, 'metadata') and doc.metadata:
                            source_info = f"{doc.metadata.get('source', 'Unknown source')}"
                            if 'page' in doc.metadata:
                                source_info += f", Page {doc.metadata['page']}"
                            sources.append(source_info)
                    
                    if sources:
                        unique_sources = list(set(sources))
                        cleaned_sources = [clean_source_text(src) for src in unique_sources]
                        source_text = "; ".join(cleaned_sources)
                        response += "\n\n📚 **Sources**: " + source_text
                else:
                    response = f"I'm sorry, but I don't have additional information about {conversation_context['last_topic']} in my knowledge base. Would you like to ask about something else?"
            except Exception as e:
                print(f"Error in follow-up handling: {str(e)}")
                response = f"I'm sorry, but I don't have additional information about {conversation_context['last_topic']} in my knowledge base. Would you like to ask about something else?"
        else:
            # The rest of your existing code for non-follow-up questions...
            # Extract the main topic from the user's message
            topic = identify_topic(user_message)
            
            # Check if the query is likely related to farming
            farming_topics = [
                "plant", "crop", "soil", "disease", "pest", "irrigation", "fertilizer", 
                "harvest", "seed", "growth", "garden", "farm", "agriculture", "cultivation",
                "organic", "compost", "weather", "season", "yield", "nutrient", "weed"
            ]
            
            is_farming_related = any(topic in user_message.lower() for topic in farming_topics)
            
            if not is_farming_related:
                # Handle non-farming topics
                non_farming_keywords = ["politics", "sports", "entertainment", "movie", "game", 
                                    "celebrity", "stock market", "music", "travel"]
                                    
                if any(keyword in user_message.lower() for keyword in non_farming_keywords):
                    response = "I'm specifically designed to help with farming and plant-related questions. For this topic, I recommend using a general-purpose assistant or a specialized tool. Can I help you with any farming or gardening questions instead?"
                else:
                    # Try to use the knowledge base directly
                    try:
                        qa_result = qa_chain(user_message)
                        response = qa_result["result"]
                        
                        # Add source attribution
                        source_docs = qa_result.get("source_documents", [])
                        if source_docs:
                            sources = []
                            for doc in source_docs:
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    source_info = f"{doc.metadata.get('source', 'Unknown source')}"
                                    if 'page' in doc.metadata:
                                        source_info += f", Page {doc.metadata['page']}"
                                    sources.append(source_info)
                            
                            if sources:
                                unique_sources = list(set(sources))
                                cleaned_sources = [clean_source_text(src) for src in unique_sources]
                                source_text = "; ".join(cleaned_sources)
                                response += "\n\n📚 **Sources**: " + source_text
                    except Exception:
                        # Fall back to agent
                        result = farming_agent.run(user_message)
                        response = result
            else:
                # For farming-related queries, use the knowledge base directly to ensure data comes from Chroma DB
                try:
                    # Always query the knowledge base directly
                    qa_result = qa_chain(user_message)
                    response = qa_result["result"]
                    
                    # Always add source attribution
                    source_docs = qa_result.get("source_documents", [])
                    if source_docs:
                        sources = []
                        for doc in source_docs:
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source_info = f"{doc.metadata.get('source', 'Unknown source')}"
                                if 'page' in doc.metadata:
                                    source_info += f", Page {doc.metadata['page']}"
                                sources.append(source_info)
                        
                        if sources:
                            unique_sources = list(set(sources))
                            cleaned_sources = [clean_source_text(src) for src in unique_sources]
                            source_text = "; ".join(cleaned_sources)
                            response += "\n\n📚 **Sources**: " + source_text
                    else:
                        # If no source documents found, try using the agent as fallback
                        augmented_message = user_message + " Please include only information from the knowledge base."
                        result = farming_agent.run(augmented_message)
                        response = result
                except Exception as e:
                    # If direct knowledge base query fails, use agent as fallback
                    try:
                        augmented_message = user_message + " Please include only information from the knowledge base with sources."
                        result = farming_agent.run(augmented_message)
                        response = result
                    except Exception:
                        response = f"I'm sorry, but I couldn't find information about this in my knowledge base. Error: {str(e)}"
                except Exception as e:
                    # If the agent fails, try the direct knowledge base approach
                    try:
                        qa_result = qa_chain(user_message)
                        response = qa_result["result"]
                        
                        # Add source attribution
                        source_docs = qa_result.get("source_documents", [])
                        if source_docs:
                            sources = []
                            for doc in source_docs:
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    source_info = f"{doc.metadata.get('source', 'Unknown source')}"
                                    if 'page' in doc.metadata:
                                        source_info += f", Page {doc.metadata['page']}"
                                    sources.append(source_info)
                            
                            if sources:
                                unique_sources = list(set(sources))
                                cleaned_sources = [clean_source_text(src) for src in unique_sources]
                                source_text = "; ".join(cleaned_sources)
                                response += "\n\n📚 **Sources**: " + source_text
                    except Exception:
                        # If all else fails, provide a generic response with the original error
                        response = f"I'm unable to process this request. There might be a technical issue with my tools or the question might be outside my expertise. Error details: {str(e)}"

            # Update the conversation context if this is not a follow-up
            if topic:
                conversation_context["last_topic"] = topic

    except Exception as e:
        # Fallback error handling
        response = "⚠️ I encountered an error while processing your request. Let me try a more direct approach."
        try:
            # Try one more time with just the knowledge base
            qa_result = qa_chain(user_message)
            response = qa_result["result"]
            
            # Add source attribution
            source_docs = qa_result.get("source_documents", [])
            if source_docs:
                sources = []
                for doc in source_docs:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source_info = f"{doc.metadata.get('source', 'Unknown source')}"
                        if 'page' in doc.metadata:
                            source_info += f", Page {doc.metadata['page']}"
                        sources.append(source_info)
                
                if sources:
                    unique_sources = list(set(sources))
                    cleaned_sources = [clean_source_text(src) for src in unique_sources]
                    source_text = "; ".join(cleaned_sources)
                    response += "\n\n📚 **Sources**: " + source_text
        except Exception:
            response = f"⚠️ I'm sorry, but I encountered an error processing your request: {str(e)}. Please try rephrasing your question or asking about a farming-related topic."
    
    # Update conversation history
    conversation_context["previous_queries"].append(user_message)
    conversation_context["previous_responses"].append(response)
    
    # Maintain maximum context length
    if len(conversation_context["previous_queries"]) > MAX_CONTEXT_ITEMS:
        conversation_context["previous_queries"] = conversation_context["previous_queries"][-MAX_CONTEXT_ITEMS:]
        conversation_context["previous_responses"] = conversation_context["previous_responses"][-MAX_CONTEXT_ITEMS:]
    
    # Update GUI history
    history.append((user_message, response))
    return history

def clear_chat():
    """
    Clear the chat history.
    
    Returns:
        list: Empty list for resetting chat history
    """
    global conversation_context
    
    # Reset conversation context
    conversation_context = {
        "last_topic": None,
        "previous_queries": [],
        "previous_responses": []
    }
    
    return []