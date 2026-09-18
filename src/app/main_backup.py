from pathlib import Path
import hashlib
import logging
import tempfile
import sys

import ollama
import streamlit as st

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Fix project path so "src" can be imported when Streamlit runs
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.document import DocumentProcessor


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_DIR = PROJECT_ROOT / "data" / "vectors"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# Generate PDF ID
# ============================================================

def generate_pdf_id(file_name, file_bytes):
    """
    Create a unique ID for each PDF.
    """

    content_hash = hashlib.md5(file_bytes).hexdigest()

    safe_name = Path(file_name).stem.replace(" ", "_")

    return f"{safe_name}_{content_hash[:10]}"


# ============================================================
# Get Ollama Models
# ============================================================

def get_available_models():
    """
    Get models installed locally in Ollama.
    """

    try:

        response = ollama.list()

        models = []

        for model in response.get("models", []):

            model_name = model.get("model")

            if model_name:
                models.append(model_name)

        return models

    except Exception as e:

        logger.error(
            f"Could not get Ollama models: {e}"
        )

        return []


# ============================================================
# Process PDF
# ============================================================

def process_pdf(file_path):
    """
    Process PDF using our custom DocumentProcessor.

    DocumentProcessor uses PyMuPDF.
    No UnstructuredPDFLoader is used.
    """

    logger.info(
        f"Processing PDF: {file_path}"
    )

    processor = DocumentProcessor(
        chunk_size=7500,
        chunk_overlap=100
    )

    # Load PDF pages
    documents = processor.load_pdf(
        file_path
    )

    if not documents:

        raise ValueError(
            "No readable text was found in this PDF."
        )

    # Split pages into chunks
    chunks = processor.split_documents(
        documents
    )

    logger.info(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# Create Chroma Vector Database
# ============================================================

def create_vector_db(
    documents,
    collection_name
):
    """
    Create a ChromaDB vector database
    using local Ollama embeddings.
    """

    logger.info(
        f"Creating vector database: {collection_name}"
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(VECTOR_DIR)
    )

    logger.info(
        "Vector database created successfully."
    )

    return vector_db


# ============================================================
# Process and Store Uploaded PDF
# ============================================================

def process_and_store_pdf(
    file_upload,
    pdf_id
):
    """
    Save uploaded PDF temporarily,
    process it with PyMuPDF,
    create embeddings,
    and store vectors in ChromaDB.
    """

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded file temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                file_upload.getvalue()
            )

            temp_path = Path(
                temp_file.name
            )

        st.info(
            f"Processing {file_upload.name}..."
        )

        # ----------------------------------------------------
        # Process PDF
        # ----------------------------------------------------

        documents = process_pdf(
            temp_path
        )

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        for document in documents:

            document.metadata[
                "pdf_id"
            ] = pdf_id

            document.metadata[
                "file_name"
            ] = file_upload.name

        st.write(
            f"Created {len(documents)} text chunks."
        )

        # ----------------------------------------------------
        # Create vector database
        # ----------------------------------------------------

        vector_db = create_vector_db(
            documents,
            pdf_id
        )

        st.success(
            f"{file_upload.name} processed successfully."
        )

        return vector_db

    except Exception as e:

        logger.exception(
            "Error while processing PDF"
        )

        st.error(
            f"Error processing {file_upload.name}: {e}"
        )

        return None

    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if (
            temp_path
            and temp_path.exists()
        ):

            try:

                temp_path.unlink()

            except Exception:

                pass


# ============================================================
# Delete One PDF
# ============================================================

def delete_pdf(pdf_id):

    try:

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

        vector_db = Chroma(
            collection_name=pdf_id,
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DIR)
        )

        vector_db.delete_collection()

        logger.info(
            f"Deleted collection: {pdf_id}"
        )

        return True

    except Exception as e:

        logger.error(
            f"Error deleting PDF: {e}"
        )

        return False


# ============================================================
# Delete All PDFs
# ============================================================

def delete_all_pdfs():

    if "pdfs" not in st.session_state:

        return

    for pdf_id in list(
        st.session_state.pdfs.keys()
    ):

        delete_pdf(pdf_id)

    st.session_state.pdfs = {}


# ============================================================
# Ask Question Across PDFs
# ============================================================

def process_question_multi_pdf(
    question,
    pdfs,
    model_name
):
    """
    Retrieve relevant chunks from uploaded PDFs
    and generate an answer using local Ollama.
    """

    if not pdfs:

        return (
            "Please upload and process a PDF first.",
            []
        )

    all_documents = []

    # --------------------------------------------------------
    # Search each PDF
    # --------------------------------------------------------

    for pdf_id, pdf_data in pdfs.items():

        vector_db = pdf_data.get(
            "vector_db"
        )

        if vector_db is None:

            continue

        try:

            retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4
                }
            )

            documents = retriever.invoke(
                question
            )

            for document in documents:

                document.metadata[
                    "pdf_id"
                ] = pdf_id

                if "file_name" not in document.metadata:

                    document.metadata[
                        "file_name"
                    ] = pdf_data.get(
                        "file_name",
                        "Unknown PDF"
                    )

                all_documents.append(
                    document
                )

        except Exception as e:

            logger.error(
                f"Error retrieving from {pdf_id}: {e}"
            )

    # --------------------------------------------------------
    # No relevant documents
    # --------------------------------------------------------

    if not all_documents:

        return (
            "The answer is not available in the provided documents.",
            []
        )

    # Use maximum 8 chunks
    all_documents = all_documents[:8]

    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in all_documents
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

If the answer is not present in the context, say:

"The answer is not available in the provided documents."

Do not invent information.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # --------------------------------------------------------
    # Local Ollama LLM
    # --------------------------------------------------------

    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    # --------------------------------------------------------
    # Create chain
    # --------------------------------------------------------

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return (
            answer,
            all_documents
        )

    except Exception as e:

        logger.exception(
            "Error generating answer"
        )

        return (
            f"Error while generating answer: {e}",
            []
        )


# ============================================================
# Main Streamlit App
# ============================================================

def main():

    # --------------------------------------------------------
    # Page configuration
    # --------------------------------------------------------

    st.set_page_config(
        page_title="Ollama PDF RAG Playground",
        page_icon="📄",
        layout="wide"
    )

    st.title(
        "Ollama PDF RAG playground"
    )

    st.markdown(
        "Upload PDF files ↓"
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "pdfs" not in st.session_state:

        st.session_state.pdfs = {}

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        st.header(
            "Settings"
        )

        # ----------------------------------------------------
        # Get local Ollama models
        # ----------------------------------------------------

        models = get_available_models()

        if not models:

            st.error(
                "No Ollama models were found."
            )

            st.info(
                "Run: ollama pull llama3.2:3b"
            )

            selected_model = None

        else:

            selected_model = st.selectbox(
                "Pick a model available locally on your system ↓",
                models
            )

        st.divider()

        # ----------------------------------------------------
        # Uploaded PDFs
        # ----------------------------------------------------

        st.subheader(
            "Uploaded PDFs"
        )

        if st.session_state.pdfs:

            for pdf_id, pdf_data in list(
                st.session_state.pdfs.items()
            ):

                file_name = pdf_data.get(
                    "file_name",
                    "Unknown PDF"
                )

                st.write(
                    f"📄 {file_name}"
                )

                if st.button(
                    f"Delete {file_name}",
                    key=f"delete_{pdf_id}"
                ):

                    delete_pdf(
                        pdf_id
                    )

                    del st.session_state.pdfs[
                        pdf_id
                    ]

                    st.rerun()

        else:

            st.write(
                "No PDFs uploaded yet."
            )

    # --------------------------------------------------------
    # Upload PDF
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --------------------------------------------------------
    # Process PDFs
    # --------------------------------------------------------

    if uploaded_files:

        for file_upload in uploaded_files:

            file_bytes = file_upload.getvalue()

            pdf_id = generate_pdf_id(
                file_upload.name,
                file_bytes
            )

            # Prevent duplicate processing
            if pdf_id not in st.session_state.pdfs:

                vector_db = process_and_store_pdf(
                    file_upload,
                    pdf_id
                )

                if vector_db is not None:

                    st.session_state.pdfs[
                        pdf_id
                    ] = {
                        "file_name": file_upload.name,
                        "vector_db": vector_db
                    }

    # --------------------------------------------------------
    # Available documents
    # --------------------------------------------------------

    if st.session_state.pdfs:

        st.divider()

        st.subheader(
            "Available documents"
        )

        for pdf_id, pdf_data in (
            st.session_state.pdfs.items()
        ):

            st.write(
                f"📄 {pdf_data.get('file_name', 'Unknown PDF')}"
            )

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Ask questions about your PDF"
    )

    # --------------------------------------------------------
    # Display previous messages
    # --------------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your uploaded PDF..."
    )

    if question:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if selected_model is None:

            st.error(
                "No local Ollama model is available."
            )

            return

        # ----------------------------------------------------
        # Check PDF
        # ----------------------------------------------------

        if not st.session_state.pdfs:

            st.warning(
                "Please upload and process a PDF first."
            )

            return

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching the PDF and generating an answer..."
            ):

                answer, source_docs = (
                    process_question_multi_pdf(
                        question,
                        st.session_state.pdfs,
                        selected_model
                    )
                )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            if source_docs:

                st.markdown(
                    "### Sources"
                )

                shown_sources = set()

                for document in source_docs:

                    file_name = document.metadata.get(
                        "file_name",
                        "Unknown PDF"
                    )

                    page_number = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    source = (
                        file_name,
                        page_number
                    )

                    if source not in shown_sources:

                        shown_sources.add(
                            source
                        )

                        st.write(
                            f"📄 {file_name} — Page {page_number}"
                        )

        # ----------------------------------------------------
        # Save assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# Start application
# ============================================================

if __name__ == "__main__":
    main()