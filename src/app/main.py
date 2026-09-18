"""
Secure PDF RAG Assistant

Features:
- Employee/Admin login
- Admin can upload company PDFs
- Employee can ask questions about company PDFs
- PDF text is read directly using PyMuPDF
- Local keyword retrieval is the primary source of truth
- Ollama generates the final answer
- ChromaDB is an optional semantic retrieval layer
- Answers include PDF sources and page numbers
- No cloud LLM API
"""

from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import json
import re
import logging
import subprocess

import streamlit as st
import pymupdf
import ollama

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Chroma is optional.
# The application will still work if Chroma has a problem.
try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_DIR = PROJECT_ROOT / "data" / "vectors"
REGISTRY_FILE = DOCUMENT_DIR / "registry.json"

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEFAULT_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_COLLECTION = "secure_pdf_rag_v4"

CHUNK_SIZE = 1800

CHUNK_OVERLAP = 150

MAX_CONTEXT_CHUNKS = 6


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Secure PDF RAG Assistant",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# USERS
# ============================================================

USERS = {
    "employee": {
        "password": "employee123",
        "role": "employee"
    },

    "admin": {
        "password": "admin123",
        "role": "admin"
    }
}


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate(
    username: str,
    password: str,
    selected_role: str
) -> bool:

    username = username.strip()
    password = password.strip()

    if username not in USERS:
        return False

    user = USERS[username]

    if user["password"] != password:
        return False

    if user["role"] != selected_role:
        return False

    return True


def show_login():

    st.title("🔐 Secure PDF RAG Assistant")

    st.write(
        "Secure company document assistant powered by local Ollama."
    )

    st.divider()

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        role = st.selectbox(
            "Login as",
            [
                "employee",
                "admin"
            ]
        )

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.button(
            "Login",
            use_container_width=True
        )

        st.caption(
            "Employee: employee / employee123"
        )

        st.caption(
            "Admin: admin / admin123"
        )

        if login:

            if authenticate(
                username,
                password,
                role
            ):

                st.session_state.authenticated = True

                st.session_state.username = username

                st.session_state.role = role

                st.session_state.chat_history = []

                st.rerun()

            else:

                st.error(
                    "Invalid username, password, or role."
                )


# ============================================================
# REGISTRY
# ============================================================

def load_registry() -> Dict:

    if not REGISTRY_FILE.exists():

        return {}

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):

            return data

        return {}

    except Exception as error:

        logger.warning(
            f"Registry error: {error}"
        )

        return {}


def save_registry(
    registry: Dict
):

    try:

        with open(
            REGISTRY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                registry,
                file,
                indent=2
            )

    except Exception as error:

        logger.error(
            f"Could not save registry: {error}"
        )


def create_pdf_id(
    file_bytes: bytes
) -> str:

    return (
        "pdf_"
        + hashlib.sha256(
            file_bytes
        ).hexdigest()[:16]
    )


# ============================================================
# SYNCHRONIZE PDF FOLDER
# ============================================================

def synchronize_documents() -> Dict:

    """
    The actual PDF folder is the source of truth.

    Even if registry.json is broken or missing,
    the application discovers PDFs directly.
    """

    registry = load_registry()

    new_registry = {}

    pdf_files = list(
        DOCUMENT_DIR.glob("*.pdf")
    )

    # --------------------------------------------------------
    # Read every actual PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        try:

            file_bytes = pdf_path.read_bytes()

            pdf_id = create_pdf_id(
                file_bytes
            )

            old_entry = registry.get(
                pdf_id,
                {}
            )

            if not isinstance(
                old_entry,
                dict
            ):
                old_entry = {}

            new_registry[pdf_id] = {

                "pdf_id": pdf_id,

                "file_name": pdf_path.name,

                "stored_path": str(
                    pdf_path
                ),

                "content_hash": pdf_id,

                "chunk_ids": old_entry.get(
                    "chunk_ids",
                    []
                ),

                "chroma_collection": old_entry.get(
                    "chroma_collection",
                    ""
                )
            }

        except Exception as error:

            logger.warning(
                f"Could not read {pdf_path}: {error}"
            )

    save_registry(
        new_registry
    )

    return new_registry


# ============================================================
# DOCUMENT SIGNATURE
# ============================================================

def get_pdf_signature(
    registry: Dict
) -> Tuple:

    signature = []

    for pdf_id, entry in sorted(
        registry.items()
    ):

        path = DOCUMENT_DIR / entry[
            "file_name"
        ]

        if not path.exists():
            continue

        try:

            stat = path.stat()

            signature.append(
                (
                    pdf_id,
                    str(path),
                    stat.st_size,
                    stat.st_mtime_ns
                )
            )

        except Exception:
            pass

    return tuple(signature)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def extract_pdf_documents(
    signature: Tuple
) -> List[Dict]:

    """
    Reads actual PDFs directly.

    Each PDF page is converted into text and then split
    into manageable chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    records = []

    for item in signature:

        pdf_id = item[0]

        pdf_path = Path(
            item[1]
        )

        if not pdf_path.exists():
            continue

        try:

            pdf = pymupdf.open(
                str(pdf_path)
            )

            for page_index in range(
                len(pdf)
            ):

                page = pdf[
                    page_index
                ]

                text = page.get_text(
                    "text"
                ).strip()

                if not text:
                    continue

                page_number = (
                    page_index + 1
                )

                document = Document(
                    page_content=text,
                    metadata={
                        "pdf_id": pdf_id,
                        "file_name": pdf_path.name,
                        "page": page_number,
                        "source": str(pdf_path)
                    }
                )

                chunks = splitter.split_documents(
                    [document]
                )

                for chunk_index, chunk in enumerate(
                    chunks
                ):

                    records.append({

                        "text": chunk.page_content,

                        "pdf_id": pdf_id,

                        "file_name": pdf_path.name,

                        "page": page_number,

                        "source": str(pdf_path),

                        "chunk_index": chunk_index
                    })

            pdf.close()

        except Exception as error:

            logger.error(
                f"PDF extraction error: {error}"
            )

    return records


def get_pdf_records(
    registry: Dict
) -> List[Dict]:

    signature = get_pdf_signature(
        registry
    )

    if not signature:

        return []

    return extract_pdf_documents(
        signature
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOP_WORDS = {
    "what",
    "what's",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "from",
    "with",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "how",
    "many",
    "much",
    "tell",
    "me",
    "please",
    "give",
    "about",
    "according",
    "there",
    "their",
    "this",
    "that",
    "it",
    "i",
    "we",
    "you",
    "your",
    "company",
    "employee",
    "employees",
    "document",
    "documents",
    "policy",
    "policies"
}


def normalize_word(
    word: str
) -> str:

    word = word.lower()

    replacements = {

        "leaves": "leave",

        "requirements": "requirement",

        "passwords": "password",

        "guidelines": "guideline",

        "policies": "policy",

        "holidays": "holiday",

        "benefits": "benefit",

        "expenses": "expense",

        "incidents": "incident",

        "requests": "request"
    }

    if word in replacements:

        return replacements[word]

    if (
        word.endswith("ies")
        and len(word) > 4
    ):

        return word[:-3] + "y"

    if (
        word.endswith("s")
        and len(word) > 3
    ):

        return word[:-1]

    return word


def tokenize(
    text: str
) -> List[str]:

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    result = []

    for word in words:

        word = normalize_word(
            word
        )

        if word in STOP_WORDS:
            continue

        if len(word) <= 1:
            continue

        result.append(
            word
        )

    return result


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_question(
    question: str
) -> str:

    q = question.lower()

    additions = []

    # Leave-related questions
    if (
        "leave" in q
        or "leaves" in q
        or "vacation" in q
        or "time off" in q
    ):

        additions.extend([
            "casual leave",
            "sick leave",
            "annual leave",
            "days per year",
            "carry forward",
            "leave application"
        ])

    # Password-related questions
    if "password" in q:

        additions.extend([
            "password requirements",
            "minimum 12 characters",
            "uppercase",
            "lowercase",
            "numbers",
            "special characters",
            "unique password",
            "personal information",
            "password manager"
        ])

    # WFH
    if (
        "work from home" in q
        or "wfh" in q
        or "remote work" in q
    ):

        additions.extend([
            "work from home",
            "WFH",
            "days per month",
            "probation",
            "manager approval",
            "employee portal",
            "working hours"
        ])

    # Phishing
    if "phishing" in q:

        additions.extend([
            "phishing awareness",
            "suspicious email",
            "unexpected message",
            "password",
            "security incident"
        ])

    # Reimbursement
    if (
        "reimbursement" in q
        or "expense" in q
    ):

        additions.extend([
            "business expense",
            "receipts",
            "manager approval",
            "30 days",
            "employee portal"
        ])

    # Holiday
    if (
        "holiday" in q
        or "holidays" in q
    ):

        additions.extend([
            "paid holidays",
            "annual calendar",
            "location",
            "employee portal"
        ])

    return (
        question
        + " "
        + " ".join(additions)
    )


# ============================================================
# DIRECT PDF SEARCH
# ============================================================

def score_record(
    question: str,
    record: Dict
) -> float:

    query_tokens = set(
        tokenize(question)
    )

    if not query_tokens:
        return 0.0

    text_tokens = set(
        tokenize(
            record["text"]
        )
    )

    if not text_tokens:
        return 0.0

    # --------------------------------------------------------
    # Main keyword overlap
    # --------------------------------------------------------

    overlap = (
        query_tokens
        & text_tokens
    )

    score = (
        len(overlap)
        / len(query_tokens)
    )

    # --------------------------------------------------------
    # Exact phrase match
    # --------------------------------------------------------

    normalized_question = " ".join(
        tokenize(question)
    )

    normalized_text = " ".join(
        tokenize(record["text"])
    )

    if (
        normalized_question
        and normalized_question
        in normalized_text
    ):

        score += 1.5

    # --------------------------------------------------------
    # Heading/title match
    # --------------------------------------------------------

    heading = " ".join(
        tokenize(
            record["text"][:400]
        )
    )

    heading_tokens = set(
        heading.split()
    )

    heading_overlap = (
        query_tokens
        & heading_tokens
    )

    if heading_overlap:

        score += (
            0.5
            * len(heading_overlap)
            / len(query_tokens)
        )

    # --------------------------------------------------------
    # Filename match
    # --------------------------------------------------------

    filename_tokens = set(
        tokenize(
            record["file_name"]
        )
    )

    filename_overlap = (
        query_tokens
        & filename_tokens
    )

    if filename_overlap:

        score += (
            0.8
            * len(filename_overlap)
            / len(query_tokens)
        )

    return score


def direct_pdf_search(
    question: str,
    records: List[Dict],
    top_k: int = 10
) -> List[Tuple[float, Dict]]:

    results = []

    for record in records:

        score = score_record(
            question,
            record
        )

        if score > 0:

            results.append(
                (
                    score,
                    record
                )
            )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# OLLAMA
# ============================================================

def get_available_models() -> List[str]:

    models = []

    # --------------------------------------------------------
    # Python Ollama package
    # --------------------------------------------------------

    try:

        response = ollama.list()

        raw_models = getattr(
            response,
            "models",
            []
        )

        for model in raw_models:

            name = getattr(
                model,
                "model",
                None
            )

            if not name:

                name = getattr(
                    model,
                    "name",
                    None
                )

            if name:

                models.append(
                    str(name)
                )

    except Exception as error:

        logger.warning(
            f"Ollama Python list failed: {error}"
        )

    # --------------------------------------------------------
    # CLI fallback
    # --------------------------------------------------------

    if not models:

        try:

            result = subprocess.run(
                [
                    "ollama",
                    "list"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:

                lines = (
                    result.stdout
                    .splitlines()
                )

                for line in lines[1:]:

                    parts = line.split()

                    if parts:

                        models.append(
                            parts[0]
                        )

        except Exception as error:

            logger.warning(
                f"Ollama CLI failed: {error}"
            )

    return sorted(
        list(
            set(models)
        )
    )


def get_chat_models() -> List[str]:

    models = get_available_models()

    return [
        model
        for model in models
        if "embed" not in model.lower()
    ]


@st.cache_resource
def get_llm(
    model_name: str
):

    return ChatOllama(
        model=model_name,
        temperature=0
    )


# ============================================================
# OPTIONAL CHROMA
# ============================================================

@st.cache_resource
def get_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


def get_chroma():

    if Chroma is None:

        return None

    try:

        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(
                VECTOR_DIR
            )
        )

    except Exception as error:

        logger.warning(
            f"Chroma unavailable: {error}"
        )

        return None


def update_chroma(
    records: List[Dict],
    registry: Dict
):

    """
    Chroma is optional.

    If it fails, the chatbot continues to use
    direct PDF retrieval.
    """

    db = get_chroma()

    if db is None:
        return

    for pdf_id, entry in registry.items():

        # Skip if already indexed
        if (
            entry.get(
                "chroma_collection"
            )
            == CHROMA_COLLECTION
            and entry.get(
                "chunk_ids"
            )
        ):
            continue

        pdf_records = [
            record
            for record in records
            if record["pdf_id"] == pdf_id
        ]

        if not pdf_records:
            continue

        try:

            documents = []

            ids = []

            for index, record in enumerate(
                pdf_records
            ):

                documents.append(
                    Document(
                        page_content=record["text"],
                        metadata={
                            "pdf_id": pdf_id,
                            "file_name": record["file_name"],
                            "page": record["page"],
                            "source": record["source"]
                        }
                    )
                )

                ids.append(
                    f"{pdf_id}_v4_{index}"
                )

            db.add_documents(
                documents=documents,
                ids=ids
            )

            registry[pdf_id][
                "chunk_ids"
            ] = ids

            registry[pdf_id][
                "chroma_collection"
            ] = CHROMA_COLLECTION

        except Exception as error:

            logger.warning(
                f"Chroma indexing skipped: {error}"
            )

    save_registry(
        registry
    )


# ============================================================
# SOURCE GROUPING
# ============================================================

def format_sources(
    records: List[Dict]
) -> str:

    grouped = {}

    for record in records:

        file_name = record[
            "file_name"
        ]

        page = record[
            "page"
        ]

        grouped.setdefault(
            file_name,
            set()
        )

        grouped[
            file_name
        ].add(
            page
        )

    lines = []

    for file_name, pages in grouped.items():

        sorted_pages = sorted(
            pages
        )

        page_text = ", ".join(
            str(page)
            for page in sorted_pages
        )

        lines.append(
            f"📄 **{file_name}** — Page(s) {page_text}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(
    records: List[Dict]
) -> str:

    parts = []

    for index, record in enumerate(
        records,
        start=1
    ):

        parts.append(
            f"""
--- SOURCE {index} ---
Document: {record["file_name"]}
Page: {record["page"]}

{record["text"]}
"""
        )

    return "\n".join(
        parts
    )


# ============================================================
# ANSWER
# ============================================================

def generate_answer(
    question: str,
    model_name: str,
    records: List[Dict]
) -> str:

    context = build_context(
        records
    )

    llm = get_llm(
        model_name
    )

    prompt = f"""
You are a secure internal company document assistant.

Your ONLY source of information is the company PDF text
provided below.

Answer the employee's question using the PDF information.

IMPORTANT:

- If the answer is present in the PDF text, ALWAYS answer it.
- Never say "the answer is not available" when the context
  contains the answer.
- Do not use outside knowledge.
- Do not invent information.
- Preserve exact numbers from the PDFs.
- If several categories are relevant, list them clearly.
- If the employee asks to summarize a policy, summarize
  the relevant policy from the PDF.
- Keep the answer concise but complete.
- Do not mention retrieval, ChromaDB, embeddings, or
  internal implementation details.

COMPANY PDF TEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content

    return answer.strip()


# ============================================================
# QUESTION PROCESSING
# ============================================================

def answer_question(
    question: str,
    model_name: str,
    records: List[Dict]
):

    question = question.strip()

    if not question:

        return (
            "Please enter a question.",
            []
        )

    # --------------------------------------------------------
    # Greeting
    # --------------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower() in greetings:

        return (
            "Hello! How can I help you with the company documents?",
            []
        )

    # --------------------------------------------------------
    # Number of PDFs
    # --------------------------------------------------------

    if (
        "how many pdf" in question.lower()
        or "how many documents" in question.lower()
    ):

        file_names = sorted(
            set(
                record["file_name"]
                for record in records
            )
        )

        if not file_names:

            return (
                "There are currently no company PDFs.",
                []
            )

        answer = (
            f"There are {len(file_names)} "
            f"company documents available:\n\n"
        )

        for index, name in enumerate(
            file_names,
            start=1
        ):

            answer += (
                f"{index}. {name}\n"
            )

        return (
            answer,
            []
        )

    # --------------------------------------------------------
    # PRIMARY RETRIEVAL:
    # ACTUAL PDF TEXT
    # --------------------------------------------------------

    direct_results = direct_pdf_search(
        question,
        records,
        top_k=10
    )

    # --------------------------------------------------------
    # If direct question has weak results,
    # use expanded question.
    # --------------------------------------------------------

    if direct_results:

        strongest_score = direct_results[0][0]

    else:

        strongest_score = 0

    if strongest_score < 0.25:

        expanded_question = expand_question(
            question
        )

        expanded_results = direct_pdf_search(
            expanded_question,
            records,
            top_k=10
        )

        # Combine results
        combined = {}

        for score, record in direct_results:

            key = (
                record["pdf_id"],
                record["page"],
                record["chunk_index"]
            )

            combined[key] = (
                score,
                record
            )

        for score, record in expanded_results:

            key = (
                record["pdf_id"],
                record["page"],
                record["chunk_index"]
            )

            if key not in combined:

                combined[key] = (
                    score * 0.75,
                    record
                )

            else:

                old_score = combined[key][0]

                combined[key] = (
                    max(
                        old_score,
                        score
                    ),
                    record
                )

        direct_results = list(
            combined.values()
        )

        direct_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

    # --------------------------------------------------------
    # Select records
    # --------------------------------------------------------

    retrieved = []

    seen = set()

    for score, record in direct_results:

        key = (
            record["pdf_id"],
            record["page"],
            record["chunk_index"]
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        retrieved.append(
            record
        )

        if len(retrieved) >= MAX_CONTEXT_CHUNKS:
            break

    # --------------------------------------------------------
    # If direct search found nothing,
    # try Chroma as a fallback.
    # --------------------------------------------------------

    if not retrieved:

        db = get_chroma()

        if db is not None:

            try:

                semantic_results = db.similarity_search(
                    expand_question(question),
                    k=6
                )

                for document in semantic_results:

                    metadata = document.metadata

                    retrieved.append({

                        "text": document.page_content,

                        "pdf_id": metadata.get(
                            "pdf_id",
                            ""
                        ),

                        "file_name": metadata.get(
                            "file_name",
                            "Unknown"
                        ),

                        "page": metadata.get(
                            "page",
                            "?"
                        ),

                        "source": metadata.get(
                            "source",
                            ""
                        ),

                        "chunk_index": 0
                    })

            except Exception as error:

                logger.warning(
                    f"Chroma search failed: {error}"
                )

    # --------------------------------------------------------
    # Still nothing
    # --------------------------------------------------------

    if not retrieved:

        return (
            "I could not find information about this question "
            "in the available company documents.",
            []
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            question,
            model_name,
            retrieved
        )

    except Exception as error:

        logger.error(
            f"Ollama error: {error}"
        )

        return (
            "I found relevant information in the company PDF, "
            "but the local Ollama model could not generate "
            "the final response.\n\n"
            f"Error: {error}",
            retrieved
        )

    # --------------------------------------------------------
    # If LLM somehow incorrectly says unavailable,
    # force one focused retry.
    # --------------------------------------------------------

    unavailable_phrases = [
        "answer is not available",
        "not available in the provided",
        "information is not available",
        "not mentioned in the provided",
        "cannot find the answer"
    ]

    if any(
        phrase in answer.lower()
        for phrase in unavailable_phrases
    ):

        try:

            focused_context = build_context(
                retrieved[:3]
            )

            llm = get_llm(
                model_name
            )

            retry_prompt = f"""
The answer to the question is contained in the company
PDF text below.

Read the text carefully and answer the question directly.

Do NOT say that the answer is unavailable.
Do NOT use outside knowledge.
Use only the PDF text.

PDF TEXT:
{focused_context}

QUESTION:
{question}

DIRECT ANSWER:
"""

            retry_response = llm.invoke(
                retry_prompt
            )

            answer = (
                retry_response.content
                .strip()
            )

        except Exception as error:

            logger.warning(
                f"Retry failed: {error}"
            )

    return (
        answer,
        retrieved
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def admin_dashboard(
    registry: Dict
):

    st.title(
        "👨‍💼 Admin Dashboard"
    )

    st.write(
        "Upload and manage company PDF documents."
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Company PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "➕ Add Selected PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please select at least one PDF."
            )

        else:

            added_count = 0

            for uploaded_file in uploaded_files:

                try:

                    file_bytes = (
                        uploaded_file.getvalue()
                    )

                    pdf_id = create_pdf_id(
                        file_bytes
                    )

                    # ----------------------------------------
                    # Duplicate check
                    # ----------------------------------------

                    if pdf_id in registry:

                        st.warning(
                            f"{uploaded_file.name} is already added."
                        )

                        continue

                    file_name = Path(
                        uploaded_file.name
                    ).name

                    target = (
                        DOCUMENT_DIR
                        / file_name
                    )

                    # ----------------------------------------
                    # Handle duplicate filename
                    # ----------------------------------------

                    if target.exists():

                        target = (
                            DOCUMENT_DIR
                            / (
                                target.stem
                                + "_"
                                + pdf_id[-6:]
                                + target.suffix
                            )
                        )

                    target.write_bytes(
                        file_bytes
                    )

                    registry[pdf_id] = {

                        "pdf_id": pdf_id,

                        "file_name": target.name,

                        "stored_path": str(
                            target
                        ),

                        "content_hash": pdf_id,

                        "chunk_ids": [],

                        "chroma_collection": ""
                    }

                    added_count += 1

                except Exception as error:

                    st.error(
                        f"Could not add "
                        f"{uploaded_file.name}: {error}"
                    )

            save_registry(
                registry
            )

            # Reset semantic indexing state
            st.session_state[
                "chroma_ready"
            ] = False

            st.success(
                f"{added_count} PDF(s) added successfully."
            )

            st.rerun()

    st.divider()

    st.subheader(
        "📚 Company Documents"
    )

    if not registry:

        st.info(
            "No company PDFs have been added."
        )

        return

    for pdf_id, entry in list(
        registry.items()
    ):

        file_name = entry.get(
            "file_name",
            "Unknown PDF"
        )

        col1, col2 = st.columns(
            [5, 1]
        )

        with col1:

            st.write(
                f"📄 **{file_name}**"
            )

            st.caption(
                "Stored locally"
            )

        with col2:

            if st.button(
                "Delete",
                key=f"delete_{pdf_id}"
            ):

                file_path = (
                    DOCUMENT_DIR
                    / file_name
                )

                # --------------------------------------------
                # Delete Chroma chunks
                # --------------------------------------------

                try:

                    db = get_chroma()

                    chunk_ids = entry.get(
                        "chunk_ids",
                        []
                    )

                    if (
                        db is not None
                        and isinstance(
                            chunk_ids,
                            list
                        )
                        and chunk_ids
                    ):

                        db.delete(
                            ids=chunk_ids
                        )

                except Exception:
                    pass

                # --------------------------------------------
                # Delete PDF
                # --------------------------------------------

                try:

                    if file_path.exists():

                        file_path.unlink()

                except Exception as error:

                    st.error(
                        f"Could not delete PDF: {error}"
                    )

                registry.pop(
                    pdf_id,
                    None
                )

                save_registry(
                    registry
                )

                st.session_state[
                    "chroma_ready"
                ] = False

                st.success(
                    f"{file_name} deleted."
                )

                st.rerun()


# ============================================================
# EMPLOYEE DASHBOARD
# ============================================================

def employee_dashboard(
    registry: Dict,
    model_name: str,
    records: List[Dict]
):

    st.title(
        "💬 Company Document Assistant"
    )

    st.write(
        "Ask questions about the authorized company PDFs."
    )

    st.divider()

    if not records:

        st.warning(
            "No company documents are currently available."
        )

        return

    # --------------------------------------------------------
    # Available documents
    # --------------------------------------------------------

    with st.expander(
        "📚 Available Company Documents"
    ):

        names = sorted(
            set(
                record["file_name"]
                for record in records
            )
        )

        for name in names:

            st.write(
                f"📄 {name}"
            )

    # --------------------------------------------------------
    # Chat history
    # --------------------------------------------------------

    if (
        "chat_history"
        not in st.session_state
    ):

        st.session_state.chat_history = []

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get(
                    "sources"
                )
            ):

                st.markdown(
                    "**Sources:**"
                )

                st.markdown(
                    format_sources(
                        message["sources"]
                    )
                )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about company policies..."
    )

    if question:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        st.session_state.chat_history.append({

            "role": "user",

            "content": question

        })

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # Assistant
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching company PDFs..."
            ):

                answer, sources = answer_question(
                    question,
                    model_name,
                    records
                )

            st.markdown(
                answer
            )

            if sources:

                st.markdown(
                    "**Sources:**"
                )

                st.markdown(
                    format_sources(
                        sources
                    )
                )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        st.session_state.chat_history.append({

            "role": "assistant",

            "content": answer,

            "sources": sources
        })


# ============================================================
# SIDEBAR
# ============================================================

def sidebar(
    available_models: List[str]
):

    with st.sidebar:

        st.header(
            "🔐 Account"
        )

        st.write(
            f"**Username:** "
            f"{st.session_state.username}"
        )

        st.write(
            f"**Role:** "
            f"{st.session_state.role}"
        )

        st.divider()

        st.subheader(
            "🤖 Local Ollama Model"
        )

        if available_models:

            default_index = 0

            if (
                DEFAULT_MODEL
                in available_models
            ):

                default_index = (
                    available_models.index(
                        DEFAULT_MODEL
                    )
                )

            selected_model = st.selectbox(
                "Select model",
                available_models,
                index=default_index
            )

        else:

            selected_model = DEFAULT_MODEL

            st.warning(
                "No Ollama chat model detected."
            )

            st.caption(
                "Make sure Ollama is running."
            )

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.authenticated = False

            st.session_state.username = ""

            st.session_state.role = ""

            st.session_state.chat_history = []

            st.rerun()

    return selected_model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if (
        "authenticated"
        not in st.session_state
    ):

        st.session_state.authenticated = False

    if (
        "username"
        not in st.session_state
    ):

        st.session_state.username = ""

    if (
        "role"
        not in st.session_state
    ):

        st.session_state.role = ""

    if (
        "chat_history"
        not in st.session_state
    ):

        st.session_state.chat_history = []

    if (
        "chroma_ready"
        not in st.session_state
    ):

        st.session_state.chroma_ready = False

    # --------------------------------------------------------
    # Login
    # --------------------------------------------------------

    if not st.session_state.authenticated:

        show_login()

        return

    # --------------------------------------------------------
    # Synchronize actual PDF folder
    # --------------------------------------------------------

    registry = synchronize_documents()

    # --------------------------------------------------------
    # Extract actual PDF text
    # --------------------------------------------------------

    records = get_pdf_records(
        registry
    )

    # --------------------------------------------------------
    # Optional Chroma preparation
    #
    # IMPORTANT:
    # This is NOT required for answering questions.
    # --------------------------------------------------------

    if (
        records
        and not st.session_state.chroma_ready
    ):

        try:

            update_chroma(
                records,
                registry
            )

        except Exception as error:

            logger.warning(
                f"Chroma setup skipped: {error}"
            )

        st.session_state.chroma_ready = True

    # --------------------------------------------------------
    # Ollama models
    # --------------------------------------------------------

    available_models = get_chat_models()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    selected_model = sidebar(
        available_models
    )

    # --------------------------------------------------------
    # Role dashboard
    # --------------------------------------------------------

    if (
        st.session_state.role
        == "admin"
    ):

        admin_dashboard(
            registry
        )

    else:

        employee_dashboard(
            registry,
            selected_model,
            records
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    main()