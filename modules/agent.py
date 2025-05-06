"""
LLM agent setup and functionality.
"""
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.tools.base import ToolException
from langchain.prompts import PromptTemplate
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain.memory import ConversationBufferMemory
from langdetect import detect
from deep_translator import GoogleTranslator
from modules.knowledge_base import setup_vector_store
from modules.disease_detector import predict_image, generate_treatment_tips
from config import GPT_CHAT_MODEL, GPT_CHAT_MODEL_LARGE, OPENAI_API_KEY

# Create a memory with a longer history
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="output"  # This helps with storing agent outputs
)

# Setup vector store and retriever
_, _, retriever = setup_vector_store()

def initialize_qa_chain(api_key=OPENAI_API_KEY):
    """
    Initialize the QA chain for knowledge retrieval.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        RetrievalQA: The initialized QA chain or None
    """
    if not api_key:
        return None
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(model=GPT_CHAT_MODEL, openai_api_key=api_key, temperature=0)
        
        # Create the QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        
        return chain
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        return None

@traceable(name="InitializeFarmingAgent", tags=["agent", "setup"])
def initialize_farming_agent(api_key=OPENAI_API_KEY):
    """
    Initialize an agent with farming-related tools.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        Agent: The initialized farming agent or None
    """
    if not api_key:
        return None
    
    try:
        # Initialize the QA chain first
        qa_chain = initialize_qa_chain(api_key)
        if not qa_chain:
            return None
        
        # Initialize the language model with minimal temperature for consistent responses
        llm = ChatOpenAI(
            model=GPT_CHAT_MODEL_LARGE,
            openai_api_key=api_key, 
            temperature=0.0,  # Use 0 temperature for consistent, deterministic responses
            max_tokens=1024
        )
        
        # Create a QA Chain with detailed prompt that emphasizes using ONLY the provided context
        qa_chain_with_prompt = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""You are a helpful farming assistant. Use ONLY the provided context to answer the question.
If the answer is not contained within the context, say: "I'm sorry, I don't have enough information about that in my knowledge base."

If this appears to be a follow-up question (e.g., "tell me more", "explain further", etc.), 
provide ONLY information from the context that wasn't covered in previous responses.
Do not generate information that is not in the context.

CONTEXT:
{context}

QUESTION:
{question}

CHAT HISTORY:
{chat_history}

ANSWER:
"""
                ),
                "verbose": True
            }
        )
        
        # Function for disease identification tool
        def identify_disease(image_path):
            try:
                result = predict_image(image_path)
                # Extract relevant information
                disease_name = result[1].replace("**Prediction: ", "").split("**")[0]
                confidence = result[1].split("(")[1].split(")")[0]
                description = result[3]
                treatment = result[4]
                
                return f"Disease: {disease_name}\nConfidence: {confidence}\nDescription: {description}\nTreatment: {treatment}"
            except Exception as e:
                raise ToolException(f"Error identifying disease: {str(e)}")
        
        # Function for knowledge base tool
        def query_knowledge_base(query):
            try:
                # Check if this is a follow-up question
                follow_up_phrases = ["tell me more", "explain more", "additional information", "continue", "elaborate"]
                is_follow_up = any(phrase in query.lower() for phrase in follow_up_phrases)
                
                # Get the memory contents
                memory_contents = memory.load_memory_variables({})
                chat_history = memory_contents.get("chat_history", "")
                
                
                # If it's a follow-up, ensure we have context from previous exchanges
                if is_follow_up and chat_history:
                    # Extract more context for the query
                    print(f"Follow-up detected: {query}")
                    
                result = qa_chain_with_prompt({
                    "query": query,
                    "chat_history": chat_history
                })
                
                response = result["result"]
                
                # Add source attribution
                source_docs = result.get("source_documents", [])
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
                        cleaned_sources = []

                        for src in unique_sources:
                            # Split name and page if exists
                            if "Page" in src:
                                name_part, page_part = src.rsplit(", Page", 1)
                            else:
                                name_part, page_part = src, ""

                            # Clean file name - remove .pdf, replace underscores and hyphens with spaces
                            name_clean = name_part.replace(".pdf", "").replace("_", " ").replace("-", " ").strip()
                            
                            # Format the source line
                            source_line = f"• {name_clean}"
                            if page_part:
                                source_line += f", Page {page_part.strip()}"
                            
                            cleaned_sources.append(source_line)

                        source_text = "\n".join(cleaned_sources)
                        response += f"\n\n📚 **Sources**:\n{source_text}"

                # Save this exchange to memory
                memory.save_context(
                    {"input": query},
                    {"output": response}
                )
                
                return response
            except Exception as e:
                raise ToolException(f"Error querying knowledge base: {str(e)}")
        
        # Function for generating treatment recommendations
        def get_treatment_recommendation(disease_name):
            try:
                return generate_treatment_tips(disease_name)
            except Exception as e:
                raise ToolException(f"Error generating treatment: {str(e)}")
        
        # Define the tools
        tools = [
            Tool(
                name="PlantDiseaseIdentifier",
                func=identify_disease,
                description="Useful for identifying plant diseases from images. Input should be a path to an image."
            ),
            Tool(
                name="FarmingKnowledgeBase",
                func=query_knowledge_base,
                description="Useful for answering questions about farming, plant care, and agriculture. Input should be a question."
            ),
            Tool(
                name="TreatmentRecommender",
                func=get_treatment_recommendation,
                description="Useful for getting treatment recommendations for plant diseases. Input should be the name of the disease."
            )
        ]
        
        # Initialize the agent with the updated memory
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        return None