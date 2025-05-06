from modules.knowledge_base import prepare_chroma_from_local_pdfs

if __name__ == "__main__":
    message = prepare_chroma_from_local_pdfs()
    print(message)
