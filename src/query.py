from sentence_transformers import SentenceTransformer
import chromadb

from llm import generate_answer

print("Loading Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Connecting To ChromaDB...")

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "documents"
)

print("RAG BOT READY")
print("Type exit to quit")

THRESHOLD = 1.2

while True:

    question = input("\nAsk Question : ")

    if question.lower() == "exit":
        break

    # Embed query
    query_embedding = model.encode(
        question
    ).tolist()

    # Retrieve documents + distances
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=8,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
    )

    best_distance = results["distances"][0][0]

    print(f"\nSimilarity Distance: {best_distance:.4f}")

    # Guardrail for unrelated questions
    if best_distance > THRESHOLD:

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(
            "I could not find this information "
            "in the provided documents."
        )

        continue

    # Build context
    context = "\n\n".join(
    results["documents"][0]
    )

    # Generate answer
    answer = generate_answer(
        question,
        context
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    shown = set()

    for meta in results["metadatas"][0]:

        source = (
            f"{meta['source']} "
            f"(Page {meta['page']})"
        )

        if source not in shown:

            print(source)

            shown.add(source)