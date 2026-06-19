from ingest import load_documents, chunk_documents
from sentence_transformers import SentenceTransformer
import chromadb

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load docs
docs = load_documents()

# Chunk docs
chunks = chunk_documents(docs)

print(f"Chunks: {len(chunks)}")

# Extract texts
texts = [chunk["text"] for chunk in chunks]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

print("Embeddings generated")

# Persistent DB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)

print("Storing in ChromaDB...")

for i, chunk in enumerate(chunks):

    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[chunk["text"]],
        metadatas=[{
            "source": chunk["source"],
            "page": chunk["page"]
        }]
    )

print("Indexing Complete")