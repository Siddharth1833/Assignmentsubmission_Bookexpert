import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 3
THRESHOLD = 1.2