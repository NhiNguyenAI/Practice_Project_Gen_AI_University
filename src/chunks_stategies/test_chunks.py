from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
# Install required libraries
# pip install langchain-core langchain-text-splitters langchain-community faiss-cpu sentence-transformers

# Sample document
document = """
Elon Musk is the CEO of Tesla. Tesla's mission is to accelerate the world's transition to sustainable energy.
Tesla manufactures electric vehicles (EVs), battery energy storage, solar panels, and related products. SpaceX, another company led by Musk,
aims to make space travel accessible to humanity by developing reusable rockets.
Musk has also co-founded Neuralink, which focuses on connecting the human brain to computers using advanced neural interfaces.
Recently, Musk has expressed his concerns about artificial intelligence, advocating for regulation to prevent potential harm.
Musk believes that governments should play an active role in ensuring the safe development of AI technologies.
His innovative ventures span automotive, space exploration, energy, AI, and neuroscience, making him a pivotal figure in modern technology.
"""

# Create text splitter
def create_text_splitter(chunk_size, chunk_overlap):
    """
    Create a text splitter with specified chunk size and overlap
    
    Args:
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

# Split document into chunks
def split_document(document, chunk_size, chunk_overlap):
    """
    Split document into chunks
    
    Args:
        document (str): Input document
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        list: List of document chunks
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Convert to Document objects
    docs = [Document(page_content=document)]
    
    # Split documents
    return splitter.split_documents(docs)

# Perform semantic search
def semantic_search(documents, query, embedding_model, k=1):
    """
    Perform semantic search on documents
    
    Args:
        documents (list): List of documents to search
        query (str): Search query
        embedding_model: Embedding model
        k (int): Number of top results to return
    
    Returns:
        list: Top matching documents
    """
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=k)
    
    return results

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Different chunk sizes
small_chunks = split_document(document, chunk_size=10, chunk_overlap=2)
large_chunks = split_document(document, chunk_size=200, chunk_overlap=10)

# Search query
query = "What are Musk's views on artificial intelligence and government regulations?"

# Perform searches
print("Small Chunks:")
for chunk in small_chunks:
    print(chunk.page_content)

print("\nLarge Chunks:")
for chunk in large_chunks:
    print(chunk.page_content)

print("\nSemantic Search Results - Small Chunks:")
small_results = semantic_search(small_chunks, query, embedding_model)
for result in small_results:
    print(result.page_content)

print("\nSemantic Search Results - Large Chunks:")
large_results = semantic_search(large_chunks, query, embedding_model)
for result in large_results:
    print(result.page_content)
