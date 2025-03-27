from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# Make sure to install these dependencies first:
# pip install langchain-community langchain-text-splitters sentence-transformers

# Small chunk size
small_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
small_chunks = small_splitter.split_text(document)
print("Smaller Chunks:")
print(small_chunks)

# Large chunk size
large_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
large_chunks = large_splitter.split_text(document)
print("\nLarger Chunks:")
print(large_chunks)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector stores
small_vector_store = Chroma.from_texts(small_chunks, embedding_model)
large_vector_store = Chroma.from_texts(large_chunks, embedding_model)

# Semantic query
query = "What are Musk's views on artificial intelligence and government regulations?"

# Perform search
small_results = small_vector_store.similarity_search(query, k=1)
large_results = large_vector_store.similarity_search(query, k=1)

print("\nSmall Chunks Search Results:")
print(small_results)
print("\nLarge Chunks Search Results:")
print(large_results)