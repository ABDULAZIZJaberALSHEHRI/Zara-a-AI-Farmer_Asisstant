"""
Knowledge base setup and document processing functionality.
"""
import os
import glob
import fitz  # PyMuPDF
from tqdm import tqdm
import chromadb
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, BOOKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Initialize embedding function
embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize Chroma client
chroma_client = chromadb.Client()

def setup_vector_store():
    """
    Setup and return the vector store.
    
    Returns:
        tuple: (collection, vectorstore, retriever) tuple
    """
    # Create or get collection
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    
    # Initialize vector store
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION_NAME, 
        embedding_function=embedding_func
    )
    
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return collection, vectorstore, retriever

def prepare_chroma_from_local_pdfs(folder_path=BOOKS_DIR, chunk_size=CHUNK_SIZE, progress=None):
    """
    Process PDF files and add them to the vector store.
    
    Args:
        folder_path: Path to the folder containing PDF files
        chunk_size: Size of text chunks for processing
        progress: Gradio progress bar object
        
    Returns:
        str: Status message
    """
    # Find PDF files
    pdf_paths = glob.glob(f"{folder_path}/*.pdf")
    if not pdf_paths:
        return f"⚠️ No PDF files found in: {folder_path}"
    
    # Get collection and vector store
    collection, vectorstore, _ = setup_vector_store()
    
    # Clear existing collection to avoid duplicates
    try:
        collection.delete(where={})
    except Exception:
        pass
    
    # Process PDF files
    all_documents = []
    print("Processing PDFs...")
    
    # Use tqdm for progress tracking
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        filename = os.path.basename(pdf_path)
        
        try:
            # Open PDF file
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    continue
                
                # Split into smaller chunks with overlap
                words = text.split()
                chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - CHUNK_OVERLAP)]
                
                # Process each chunk
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    
                    # Create document with metadata
                    document = Document(
                        page_content=chunk_text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "chunk": i
                        }
                    )
                    all_documents.append(document)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Add documents in batches to avoid memory issues
    batch_size = 100
    batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]
    
    print("Adding documents to vector store...")
    for batch in tqdm(batches, desc="Adding to vector store"):
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error adding batch: {str(e)}")
    
    return f"✅ Vector store populated with {len(all_documents)} chunks from {len(pdf_paths)} PDF files"