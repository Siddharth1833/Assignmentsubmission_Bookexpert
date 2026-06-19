# src/ingest.py

from pypdf import PdfReader
import os

DATA_FOLDER = "data"


def load_documents():
    documents = []

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".pdf"):

            path = os.path.join(DATA_FOLDER, file)

            reader = PdfReader(path)

            for page_num, page in enumerate(reader.pages):

                text = page.extract_text()

                if text:

                    documents.append({
                        "text": text,
                        "source": file,
                        "page": page_num + 1
                    })

    return documents

def chunk_documents(documents,
                    chunk_size=500,
                    overlap=100):

    chunks = []

    for doc in documents:

        text = doc["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"]
            })

            start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    docs = load_documents()

    print(f"Loaded {len(docs)} pages")

    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print(chunks[0])