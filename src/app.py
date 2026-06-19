import streamlit as st
import chromadb
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from src.llm import generate_answer

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Document Intelligence Hub",
    page_icon="📚",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.main-title {
    text-align:center;
    font-size:52px;
    font-weight:700;
    color:white;
}

.subtitle {
    text-align:center;
    color:#B3B8C5;
    font-size:18px;
    margin-bottom:30px;
}

.chat-card {
    background:#1A1D24;
    padding:18px;
    border-radius:15px;
    border:1px solid #2A2F3A;
    margin-bottom:15px;
}

.source-card {
    background:#15181E;
    padding:10px;
    border-radius:10px;
    border:1px solid #2A2F3A;
    margin-bottom:5px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    return client.get_collection(
        "documents"
    )

model = load_model()
collection = load_collection()

# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("📚 RAG Assistant")

    st.markdown("---")

    st.metric(
        "Documents",
        "5"
    )

    st.metric(
        "Chunks",
        "3650"
    )

    st.metric(
        "Vector DB",
        "ChromaDB"
    )

    st.markdown("---")

    top_k = st.slider(
        "Retrieved Chunks",
        1,
        10,
        5
    )

    threshold = st.slider(
        "Similarity Threshold",
        0.5,
        2.0,
        1.2
    )

    st.markdown("---")

    st.markdown("""
### Tech Stack

- Sentence Transformers
- ChromaDB
- Groq
- Streamlit
""")

# ----------------------------------
# SESSION STATE
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    """
<div class="main-title">
📚 Document Intelligence Hub
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
Ask questions across thousands of document pages instantly.
</div>
""",
    unsafe_allow_html=True
)

# ----------------------------------
# CHAT HISTORY
# ----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "sources" in message:

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:
                    st.write(source)

# ----------------------------------
# CHAT INPUT
# ----------------------------------

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            query_embedding = model.encode(
                question
            ).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            best_distance = (
                results["distances"][0][0]
            )

            if best_distance > threshold:

                answer = (
                    "I could not find this information "
                    "in the provided documents."
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            else:

                context = "\n\n".join(
                    results["documents"][0]
                )

                answer = generate_answer(
                    question,
                    context
                )

                st.markdown(answer)

                sources = []

                for meta in results["metadatas"][0]:

                    source = (
                        f"{meta['source']} "
                        f"(Page {meta['page']})"
                    )

                    if source not in sources:
                        sources.append(source)

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:
                        st.write(source)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )